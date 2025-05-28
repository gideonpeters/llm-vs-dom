# Use a slim Python base
FROM python:3.10-slim

# 1) Install OS dependencies, Node.js, Chromium
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       curl gnupg ca-certificates \
    && curl -fsSL https://deb.nodesource.com/setup_16.x | bash - \
    && apt-get install -y --no-install-recommends \
       nodejs \
       chromium \
    && rm -rf /var/lib/apt/lists/*

# 2) Install Lighthouse globally
RUN npm install -g lighthouse

# 3) Tell Lighthouse where Chrome is
ENV CHROME_PATH=/usr/bin/chromium

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# copy everything under scripts/ into /app/scripts
COPY scripts/ ./scripts/
# ensure the script is runnable
RUN chmod +x ./scripts/audits_reporter.py

# create results dir
RUN mkdir -p /app/results

# default entrypoint runs your converted script
CMD ["python", "scripts/audits_reporter.py"]