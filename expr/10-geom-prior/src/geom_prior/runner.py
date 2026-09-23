import argparse

import torch
import torch.nn.functional as F
import torch.optim as optim
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer


# =====================================================================
# STEP 1: DEFINE THE ADAPTIVE CLOSED-LOOP STEERING HOOK
# =====================================================================
class ClosedLoopAdaptiveSteeringHook:
    def __init__(
        self,
        projection_matrix,
        spline_coefs,
        lm_head_weight,
        base_alpha=0.5,
        entropy_threshold=2.0,
        max_alpha=0.8,
        max_update_ratio=0.05,
        goal_blend=0.1,
        adaptive_steering=True,
        tokenizer=None,
        stream_token_comparison=False,
    ):
        """
        Calculates analytical transport gradients and dynamically scales steering force (alpha)
        and adjusts steering direction if token-generation entropy spikes.
        """
        device = projection_matrix.device
        self.U = projection_matrix.to(device)  # [3072, K]
        self.a = spline_coefs["a"].to(device)  # [K]
        self.b = spline_coefs["b"].to(device)  # [K]
        self.c = spline_coefs["c"].to(device)  # [K]
        self.W_head = lm_head_weight.to(device)  # [Vocab_Size, 3072]
        self.base_alpha = base_alpha
        self.entropy_threshold = entropy_threshold
        self.max_alpha = max_alpha
        self.max_update_ratio = max_update_ratio
        self.goal_blend = goal_blend
        self.adaptive_steering = adaptive_steering
        self.tokenizer = tokenizer
        self.stream_token_comparison = stream_token_comparison
        self.generation_step = 0

        if adaptive_steering and base_alpha > max_alpha:
            raise ValueError("base_alpha cannot be greater than max_alpha.")
        if stream_token_comparison and tokenizer is None:
            raise ValueError(
                "A tokenizer is required when stream_token_comparison is enabled."
            )

    def __call__(self, module, inputs, outputs):
        output_is_tuple = isinstance(outputs, tuple)
        hidden_states = outputs[0] if output_is_tuple else outputs
        hidden_states = hidden_states.clone()
        x_raw = hidden_states[:, -1, :]  # Hidden state of the last token: [batch, 3072]

        # 1. Compute current token logits and instantaneous Shannon Entropy
        logits = torch.matmul(x_raw, self.W_head.t())  # [batch, Vocab_Size]
        probs = F.softmax(logits.float(), dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=-1)  # [batch]

        # 2. Project down to latent space for the standard Spline Gradient
        z = torch.matmul(x_raw, self.U)
        T = self.a * (z**2) + self.b * z + self.c
        dT = 2 * self.a * z + self.b
        ddT = 2 * self.a
        min_magnitude = 1e-4
        signed_min = torch.where(
            dT < 0,
            torch.full_like(dT, -min_magnitude),
            torch.full_like(dT, min_magnitude),
        )
        dT_stable = torch.where(dT.abs() < min_magnitude, signed_min, dT)

        # Baseline manifold-following steering vector
        v_latent = -(T * dT) + (ddT / dT_stable)
        v_spline_steer = torch.matmul(v_latent, self.U.t())  # [batch, 3072]

        # 3. Dynamic Alpha Modulation
        if self.adaptive_steering:
            steering_alpha = torch.clamp(
                self.base_alpha
                * (1.0 + torch.relu(entropy - self.entropy_threshold)),
                max=self.max_alpha,
            )
        else:
            steering_alpha = torch.full_like(entropy, self.base_alpha)

        # 4. Dynamic Goal Shifting (Triggered if entropy crosses the threshold)
        final_v_steer = v_spline_steer.clone()

        for i in range(x_raw.shape[0]):
            if (
                self.adaptive_steering
                and entropy[i] > self.entropy_threshold
            ):
                # Find the top 5 competing tokens causing high entropy
                top_probs, top_indices = torch.topk(probs[i], k=min(5, probs.shape[-1]))
                competing_embeddings = self.W_head[top_indices].float()

                # Calculate a consensus goal vector from competing branches weighted by probability
                weighted_goal = torch.sum(
                    competing_embeddings * top_probs.unsqueeze(-1), dim=0
                )
                v_local_correction = weighted_goal.to(x_raw.dtype)
                spline_norm = torch.linalg.vector_norm(v_spline_steer[i])
                local_norm = torch.linalg.vector_norm(v_local_correction)
                v_local_correction = v_local_correction * (
                    spline_norm / local_norm.clamp_min(1e-6)
                )

                # Shift the goal by blending the spline manifold vector with local correction
                final_v_steer[i] = (
                    v_spline_steer[i] + self.goal_blend * v_local_correction
                )

        # 5. Inject the adaptive, closed-loop corrected hidden state back into the layer
        alpha_broadcast = steering_alpha.unsqueeze(-1).to(x_raw.dtype)
        steering_update = alpha_broadcast * final_v_steer
        update_norm = torch.linalg.vector_norm(
            steering_update.float(), dim=-1, keepdim=True
        )
        max_update_norm = self.max_update_ratio * torch.linalg.vector_norm(
            x_raw.float(), dim=-1, keepdim=True
        )
        update_scale = torch.clamp(
            max_update_norm / update_norm.clamp_min(1e-6), max=1.0
        ).to(x_raw.dtype)
        hidden_states[:, -1, :] = x_raw + update_scale * steering_update

        if self.stream_token_comparison:
            updated_logits = torch.matmul(
                hidden_states[:, -1, :], self.W_head.t()
            )
            original_token_ids = torch.argmax(logits, dim=-1)
            updated_token_ids = torch.argmax(updated_logits, dim=-1)
            self.generation_step += 1
            for batch_index, (original_id, updated_id) in enumerate(
                zip(original_token_ids.tolist(), updated_token_ids.tolist())
            ):
                original_token = self.tokenizer.decode(
                    [original_id], clean_up_tokenization_spaces=False
                )
                updated_token = self.tokenizer.decode(
                    [updated_id], clean_up_tokenization_spaces=False
                )
                batch_label = (
                    f" batch={batch_index}" if len(original_token_ids) > 1 else ""
                )
                print(
                    f"[step {self.generation_step:03d}{batch_label}] "
                    f"original={original_token!r} steered={updated_token!r}",
                    flush=True,
                )

        if output_is_tuple:
            return (hidden_states,) + outputs[1:]
        return hidden_states


# =====================================================================
# STEP 2: LOW-ENTROPY SAMPLE COLLECTION PIPELINE
# =====================================================================
def collect_low_entropy_samples(
    model,
    tokenizer,
    prompt,
    num_samples=64,
    layer_idx=16,
    optimization_steps=40,
    learning_rate=0.01,
    noise_scale=0.05,
    prior_weight=5.0,
):
    """
    Captures the prompt activation prior, uses Adam optimization to discover nearby
    low-entropy active regions, and collects 64 non-linear samples.
    """
    model.eval()
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    if num_samples < 2 or num_samples % 2:
        raise ValueError("num_samples must be an even integer of at least 2.")
    if not 0 <= layer_idx < len(model.model.layers):
        raise ValueError(
            f"layer_idx must be between 0 and {len(model.model.layers) - 1}."
        )

    hidden_storage = []

    def capture_hook(module, inp, out):
        hidden_states = out[0] if isinstance(out, tuple) else out
        hidden_storage.append(hidden_states[:, -1, :].clone().detach())

    hook_handle = model.model.layers[layer_idx].register_forward_hook(capture_hook)
    try:
        with torch.no_grad():
            _ = model(**inputs)
    finally:
        hook_handle.remove()

    x_prior = hidden_storage[0].float()  # [1, 3072]
    hidden_dim = x_prior.shape[-1]

    # Isotropic initialization around the prior vector
    noise = torch.randn(num_samples, hidden_dim, device=x_prior.device) * noise_scale
    x_candidates = (
        (x_prior.repeat(num_samples, 1) + noise).detach().requires_grad_(True)
    )
    optimizer = optim.Adam([x_candidates], lr=learning_rate)

    print(
        f"\n[1/4] Optimizing {num_samples} candidates toward nearby "
        "low-entropy regions..."
    )
    for _ in range(optimization_steps):
        optimizer.zero_grad()
        logits = model.lm_head(x_candidates.to(model.dtype))
        probs = F.softmax(logits.float(), dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=-1)

        loss_entropy = torch.mean(entropy)
        loss_prior = F.mse_loss(x_candidates, x_prior.repeat(num_samples, 1))
        total_loss = loss_entropy + prior_weight * loss_prior

        total_loss.backward()
        optimizer.step()

    return x_candidates.detach().to(model.dtype)


# =====================================================================
# STEP 2B: CONTRASTIVE TEXT ACTIVATION COLLECTION
# =====================================================================
def collect_contrastive_text_activations(
    model,
    tokenizer,
    target_texts,
    opposite_texts,
    layer_idx=16,
):
    """Capture paired target and opposite activations at the selected layer."""
    if not target_texts or len(target_texts) != len(opposite_texts):
        raise ValueError(
            "target_texts and opposite_texts must contain the same non-zero "
            "number of entries."
        )
    if not 0 <= layer_idx < len(model.model.layers):
        raise ValueError(
            f"layer_idx must be between 0 and {len(model.model.layers) - 1}."
        )

    model.eval()
    hidden_storage = []

    def capture_hook(module, inp, out):
        hidden_states = out[0] if isinstance(out, tuple) else out
        hidden_storage.append(hidden_states[:, -1, :].detach())

    hook_handle = model.model.layers[layer_idx].register_forward_hook(capture_hook)
    try:
        with torch.no_grad():
            for text in [*target_texts, *opposite_texts]:
                inputs = tokenizer(text, return_tensors="pt").to(model.device)
                _ = model(**inputs)
    finally:
        hook_handle.remove()

    activations = torch.cat(hidden_storage, dim=0).to(model.dtype)
    pair_count = len(target_texts)
    return activations[:pair_count], activations[pair_count:]


# =====================================================================
# STEP 3: CONTRASTIVE SVD SUBSPACE SEPARATION (U MATRIX)
# =====================================================================
def compute_orthogonal_subspace(model, samples_64, K=2):
    """
    Ranks the 64 samples by their true Shannon entropy, splits them 32/32,
    and applies SVD to extract the orthogonal concept projection matrix U.
    """
    print("\n[2/4] Performing contrastive entropy split and SVD reduction...")
    with torch.no_grad():
        logits = model.lm_head(samples_64)
        probs = F.softmax(logits.float(), dim=-1)
        entropies = -torch.sum(probs * torch.log(probs + 1e-9), dim=-1)

    if samples_64.shape[0] < 2 or samples_64.shape[0] % 2:
        raise ValueError("The sample count must be an even integer of at least 2.")

    midpoint = samples_64.shape[0] // 2
    sorted_indices = torch.argsort(entropies)
    X_target = samples_64[sorted_indices[:midpoint]]
    X_contrast = samples_64[sorted_indices[midpoint:]]

    X_delta = X_target - X_contrast
    if not 1 <= K <= min(X_delta.shape):
        raise ValueError(f"K must be between 1 and {min(X_delta.shape)}.")

    _, _, Vh = torch.linalg.svd(
        X_delta.to(device="cpu", dtype=torch.float32), full_matrices=False
    )
    U = Vh[:K, :].t()  # Column-orthogonal projection matrix [3072, K]

    return U.to(device=samples_64.device, dtype=samples_64.dtype), X_target


def compute_contrastive_text_subspace(target_activations, opposite_activations, K=2):
    """Extract a steering subspace from paired target-minus-opposite activations."""
    if target_activations.shape != opposite_activations.shape:
        raise ValueError("Target and opposite activations must have matching shapes.")

    activation_deltas = target_activations - opposite_activations
    if not 1 <= K <= min(activation_deltas.shape):
        raise ValueError(
            f"K must be between 1 and {min(activation_deltas.shape)} for "
            f"{activation_deltas.shape[0]} text pair(s)."
        )

    _, _, Vh = torch.linalg.svd(
        activation_deltas.to(device="cpu", dtype=torch.float32),
        full_matrices=False,
    )
    projection_matrix = Vh[:K, :].t()
    return projection_matrix.to(
        device=target_activations.device,
        dtype=target_activations.dtype,
    )


# =====================================================================
# STEP 4: MONOTONIC TRANSPORT SPLINE CALIBRATION
# =====================================================================
def calibrate_transport_splines(U_matrix, X_target, steps=150):
    """
    Fits continuous, monotonic quadratic spline coefficients (a, b, c)
    independently across the K latent dimensions using a Maximum Likelihood Objective.
    """
    print("\n[3/4] Calibrating monotonic quadratic transport splines...")
    z_targets = torch.matmul(X_target.float(), U_matrix.float())
    K = z_targets.shape[-1]

    device = z_targets.device
    a = torch.zeros(K, device=device, requires_grad=True)
    raw_b = torch.zeros(K, device=device, requires_grad=True)
    c = torch.zeros(K, device=device, requires_grad=True)
    optimizer = optim.Adam([a, raw_b, c], lr=0.01)
    min_derivative = 1e-4

    for _ in range(steps):
        optimizer.zero_grad()
        loss = torch.zeros((), device=device)
        for k in range(K):
            zk = z_targets[:, k]
            derivative_without_intercept = 2 * a[k] * zk
            b_k = (
                F.softplus(raw_b[k])
                + min_derivative
                - derivative_without_intercept.min()
            )
            T_zk = a[k] * (zk**2) + b_k * zk + c[k]
            dT_zk = derivative_without_intercept + b_k

            gaussian_loss = 0.5 * torch.mean(T_zk**2)
            jacobian_loss = -torch.mean(torch.log(dT_zk))
            l2_reg = 0.01 * (a[k] ** 2 + (b_k - 1.0) ** 2 + c[k] ** 2)

            loss = loss + gaussian_loss + jacobian_loss + l2_reg

        loss.backward()
        optimizer.step()

    b = torch.stack(
        [
            F.softplus(raw_b[k]) + min_derivative - (2 * a[k] * z_targets[:, k]).min()
            for k in range(K)
        ]
    )
    coefficient_dtype = U_matrix.dtype
    return {
        "a": a.detach().to(coefficient_dtype),
        "b": b.detach().to(coefficient_dtype),
        "c": c.detach().to(coefficient_dtype),
    }


# =====================================================================
# STEP 5: MAIN EXECUTION AND RUNTIME TEST
# =====================================================================
def positive_int(value):
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def even_int(value):
    parsed = positive_int(value)
    if parsed % 2:
        raise argparse.ArgumentTypeError("must be even")
    return parsed


def positive_float(value):
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def nonnegative_float(value):
    parsed = float(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be zero or greater")
    return parsed


def format_prompt(tokenizer, prompt, use_chat_template, enable_reasoning=True):
    if use_chat_template is False:
        if not enable_reasoning:
            raise ValueError("--no-reasoning requires a tokenizer chat template.")
        return prompt
    if tokenizer.chat_template is None:
        if use_chat_template or not enable_reasoning:
            raise ValueError(
                "The tokenizer has no chat template, so reasoning mode cannot "
                "be configured."
            )
        return prompt
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_reasoning,
    )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Run adaptive activation steering on SmolLM3."
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Torch device to use (default: auto; examples: cpu, mps, cuda, cuda:1).",
    )
    parser.add_argument(
        "--model-id",
        default="HuggingFaceTB/SmolLM3-3B",
        help="Hugging Face model ID or local model path.",
    )
    parser.add_argument(
        "--prior-prompt",
        default=(
            "Provide a rigorously logical, factual breakdown of cryptography "
            "principles."
        ),
        help=(
            "Prompt used to dynamically derive the steering manifold when "
            "contrastive text pairs are not supplied."
        ),
    )
    parser.add_argument(
        "--steering-text",
        nargs="+",
        default=None,
        metavar="TEXT",
        help=(
            "Desired texts used to derive the steering subspace. Supply the same "
            "number of entries as --steering-opposite."
        ),
    )
    parser.add_argument(
        "--steering-opposite",
        nargs="+",
        default=None,
        metavar="TEXT",
        help=(
            "Opposite texts paired by position with --steering-text. Providing "
            "contrastive pairs replaces dynamic low-entropy sample discovery."
        ),
    )
    parser.add_argument(
        "--prompt",
        default="The ultimate fundamental rule of data encryption security is that",
        help="Prompt from which to generate text.",
    )
    parser.add_argument(
        "--chat-template",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Use the tokenizer's instruction chat template. By default it is "
            "used when the tokenizer provides one."
        ),
    )
    parser.add_argument(
        "--reasoning",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Enable SmolLM3 extended thinking; use --no-reasoning to disable it "
            "(default: on)."
        ),
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=16,
        help="Transformer layer to capture and steer (default: 16).",
    )
    parser.add_argument(
        "--latent-dim",
        type=positive_int,
        default=2,
        help="Number of SVD steering dimensions (default: 2).",
    )
    parser.add_argument(
        "--num-samples",
        type=even_int,
        default=64,
        help="Even number of activation candidates (default: 64).",
    )
    parser.add_argument(
        "--sample-steps",
        type=positive_int,
        default=40,
        help="Optimization steps for activation candidates (default: 40).",
    )
    parser.add_argument(
        "--sample-learning-rate",
        type=positive_float,
        default=0.01,
        help="Candidate optimization learning rate (default: 0.01).",
    )
    parser.add_argument(
        "--noise-scale",
        type=nonnegative_float,
        default=0.05,
        help="Standard deviation of candidate initialization noise (default: 0.05).",
    )
    parser.add_argument(
        "--prior-weight",
        type=nonnegative_float,
        default=5.0,
        help="Weight keeping candidates near the prior activation (default: 5.0).",
    )
    parser.add_argument(
        "--calibration-steps",
        type=positive_int,
        default=150,
        help="Spline calibration steps (default: 150).",
    )
    parser.add_argument(
        "--alpha",
        type=nonnegative_float,
        default=0.6,
        help="Base steering strength (default: 0.6).",
    )
    parser.add_argument(
        "--steering",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable activation steering; use --no-steering for an unsteered baseline.",
    )
    parser.add_argument(
        "--adaptive-steering",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Adapt steering strength and direction based on token entropy; use "
            "--no-adaptive-steering for fixed-alpha geometric steering (default: on)."
        ),
    )
    parser.add_argument(
        "--entropy-threshold",
        type=nonnegative_float,
        default=4.0,
        help="Entropy threshold for adaptive correction (default: 4.0).",
    )
    parser.add_argument(
        "--max-alpha",
        type=positive_float,
        default=0.8,
        help="Upper bound for adaptive steering strength (default: 0.8).",
    )
    parser.add_argument(
        "--max-update-ratio",
        type=nonnegative_float,
        default=0.05,
        help=(
            "Maximum update norm as a fraction of hidden-state norm "
            "(default: 0.05)."
        ),
    )
    parser.add_argument(
        "--goal-blend",
        type=nonnegative_float,
        default=0.1,
        help="Weight of the entropy-triggered local correction (default: 0.1).",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=positive_int,
        default=40,
        help="Maximum number of tokens to generate (default: 40).",
    )
    parser.add_argument(
        "--temperature",
        type=positive_float,
        default=0.8,
        help="Sampling temperature (default: 0.8).",
    )
    parser.add_argument(
        "--do-sample",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Sample tokens; use --no-do-sample for greedy decoding (default: on).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible candidate optimization and generation.",
    )
    parser.add_argument(
        "--stream-token-comparison",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Print the layer-level top token before and after steering at each "
            "generation step (default: on)."
        ),
    )
    parser.add_argument(
        "--stream-output",
        action="store_true",
        help=(
            "Stream actual generated text instead of token-comparison proxies, "
            "without printing the output again at the end."
        ),
    )
    return parser.parse_args(argv)


def resolve_device(requested_device):
    if requested_device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    device = torch.device(requested_device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but no CUDA device is available.")
    if (
        device.type == "cuda"
        and device.index is not None
        and device.index >= torch.cuda.device_count()
    ):
        raise RuntimeError(
            f"CUDA device {device.index} was requested, but only "
            f"{torch.cuda.device_count()} CUDA device(s) are available."
        )
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested, but MPS is not available.")
    return device


def dtype_for_device(device):
    if device.type == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if device.type == "mps":
        return torch.float16
    return torch.float32


def main():
    args = parse_args()
    device = resolve_device(args.device)
    model_dtype = dtype_for_device(device)
    if (args.steering_text is None) != (args.steering_opposite is None):
        raise ValueError(
            "--steering-text and --steering-opposite must be provided together."
        )
    if (
        args.steering_text is not None
        and len(args.steering_text) != len(args.steering_opposite)
    ):
        raise ValueError(
            "--steering-text and --steering-opposite must have the same number "
            "of entries."
        )
    if (
        args.steering_text is not None
        and args.latent_dim > len(args.steering_text)
    ):
        raise ValueError(
            "--latent-dim cannot exceed the number of contrastive text pairs."
        )
    if args.steering and args.adaptive_steering and args.alpha > args.max_alpha:
        raise ValueError("--alpha cannot be greater than --max-alpha.")
    if args.seed is not None:
        torch.manual_seed(args.seed)

    print(
        f"Initializing setup. Loading {args.model_id} onto {device} "
        f"with {model_dtype}..."
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    model = AutoModelForCausalLM.from_pretrained(args.model_id, dtype=model_dtype).to(
        device
    )
    stream_token_comparison = (
        args.steering and args.stream_token_comparison and not args.stream_output
    )
    hook_handle = None
    if args.steering:
        if args.steering_text is not None:
            target_texts = [
                format_prompt(
                    tokenizer,
                    text,
                    args.chat_template,
                    enable_reasoning=args.reasoning,
                )
                for text in args.steering_text
            ]
            opposite_texts = [
                format_prompt(
                    tokenizer,
                    text,
                    args.chat_template,
                    enable_reasoning=args.reasoning,
                )
                for text in args.steering_opposite
            ]
            print(
                f"\n[1/4] Capturing {len(target_texts)} contrastive text pair(s)..."
            )
            X_target, X_contrast = collect_contrastive_text_activations(
                model,
                tokenizer,
                target_texts,
                opposite_texts,
                layer_idx=args.layer,
            )
            print("\n[2/4] Computing contrastive text steering subspace...")
            U_matrix = compute_contrastive_text_subspace(
                X_target,
                X_contrast,
                K=args.latent_dim,
            )
        else:
            prior_prompt = format_prompt(
                tokenizer,
                args.prior_prompt,
                args.chat_template,
                enable_reasoning=args.reasoning,
            )
            samples = collect_low_entropy_samples(
                model,
                tokenizer,
                prior_prompt,
                num_samples=args.num_samples,
                layer_idx=args.layer,
                optimization_steps=args.sample_steps,
                learning_rate=args.sample_learning_rate,
                noise_scale=args.noise_scale,
                prior_weight=args.prior_weight,
            )
            U_matrix, X_target = compute_orthogonal_subspace(
                model, samples, K=args.latent_dim
            )
        spline_coefficients = calibrate_transport_splines(
            U_matrix, X_target, steps=args.calibration_steps
        )
        closed_loop_steer = ClosedLoopAdaptiveSteeringHook(
            projection_matrix=U_matrix,
            spline_coefs=spline_coefficients,
            lm_head_weight=model.lm_head.weight.data,
            base_alpha=args.alpha,
            entropy_threshold=args.entropy_threshold,
            max_alpha=args.max_alpha,
            max_update_ratio=args.max_update_ratio,
            goal_blend=args.goal_blend,
            adaptive_steering=args.adaptive_steering,
            tokenizer=tokenizer,
            stream_token_comparison=stream_token_comparison,
        )
        hook_handle = model.model.layers[args.layer].register_forward_hook(
            closed_loop_steer
        )

    try:
        if args.steering:
            steering_mode = "adaptive" if args.adaptive_steering else "fixed-alpha"
            print(
                f"\n[4/4] {steering_mode.capitalize()} steering controller deployed."
            )
            print(
                "----------------------------------------------------------------------------"
            )
            if stream_token_comparison:
                print(
                    "Streaming layer-level token proxies; "
                    "the sampled output may differ."
                )
            print(f"Generating text with {steering_mode} steering...\n")
        else:
            print("\nGenerating text without steering...\n")

        prompt = format_prompt(
            tokenizer,
            args.prompt,
            args.chat_template,
            enable_reasoning=args.reasoning,
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        generation_options = {
            "max_new_tokens": args.max_new_tokens,
            "do_sample": args.do_sample,
        }
        if args.do_sample:
            generation_options["temperature"] = args.temperature
        if args.stream_output:
            generation_options["streamer"] = TextStreamer(
                tokenizer,
                skip_prompt=True,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
        outputs = model.generate(**inputs, **generation_options)
        if not args.stream_output:
            generated_ids = outputs[0, inputs["input_ids"].shape[1] :]
            print(
                tokenizer.decode(
                    generated_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
            )
    finally:
        if hook_handle is not None:
            hook_handle.remove()
            print("\nHook detached successfully.")


if __name__ == "__main__":
    main()
