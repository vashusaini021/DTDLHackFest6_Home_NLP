FROM ubuntu:22.04


# Avoid tzdata prompt
ENV DEBIAN_FRONTEND=noninteractive

# Install dependencies
RUN apt update && apt install -y \
    git curl make build-essential libssl-dev zlib1g-dev \
    libbz2-dev libreadline-dev libsqlite3-dev wget \
    llvm libncursesw5-dev xz-utils tk-dev libxml2-dev \
    libxmlsec1-dev libffi-dev liblzma-dev \
    tzdata \
    && rm -rf /var/lib/apt/lists/*

# Install pyenv
ENV PYENV_ROOT="/root/.pyenv"
ENV PATH="$PYENV_ROOT/bin:$PYENV_ROOT/shims:$PATH"

RUN git clone https://github.com/pyenv/pyenv.git $PYENV_ROOT \
    && pyenv install 3.13.5 \
    && pyenv global 3.13.5

# Set working dir
WORKDIR /hackfest6

# Copy project and install requirements
COPY . .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Run app (adjust to your framework)
CMD ["python", "app.py"]
