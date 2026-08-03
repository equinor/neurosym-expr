FROM nvcr.io/nvidia/pytorch:26.07-py3
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

RUN apt-get update


COPY . /package
WORKDIR /package

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync

ENV PATH="/package/.venv/bin:$PATH"

CMD ["uv", "run", "python"]
