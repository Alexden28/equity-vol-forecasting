# Walk-forward forecasts on CPU (the LSTM is slower than on a GPU; add --no-lstm to skip it).
#   docker build -t volfc .
#   docker run --rm -v "$PWD/reports:/app/reports" volfc run --index NASDAQCOM
#   docker run --rm --entrypoint pytest volfc -q
FROM python:3.12-slim
WORKDIR /app
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
COPY pyproject.toml README.md LICENSE ./
COPY volfc volfc
COPY tests tests
RUN pip install --no-cache-dir -e ".[lstm,dev]"
ENTRYPOINT ["python", "-m", "volfc"]
CMD ["run", "--index", "NASDAQCOM"]
