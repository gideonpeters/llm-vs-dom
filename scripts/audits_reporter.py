#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import subprocess
import json
import logging
from datetime import datetime

import http.server
import socketserver
import threading
import time
import requests

# In[6]:


logging.basicConfig(filename='error.log', level=logging.ERROR, 
                    format='%(asctime)s:%(levelname)s:%(message)s')

port = 8080


# In[7]:


def run_lighthouse(url, result_path):
    try:

        print(f"Running Lighthouse for {url}")

        print(f"Saving report to {result_path}")

        # Run the Lighthouse CLI command on the HTML file
        command = [
            "lighthouse",
            url,
            "--output=json",
            "--output-path=" + result_path,
            "--chrome-flags=--headless  --no-sandbox --disable-gpu",
            "--only-categories=performance"
        ]
        subprocess.run(command, check=True)
        
        # Read the Lighthouse report
        with open(result_path, 'r') as file:
            report = json.load(file)
        
        # Extract scores
        scores = {
            'performance': report['categories']['performance']['score'] * 100
        }
        
        return scores
    except Exception as e:
        logging.error(f"{url} --- An error occurred: {e}")
        return None


# In[8]:


# path_to_dom_trees = "./../dataset/original/"
path_to_dom_trees = "./../dataset/original/"

def get_html_files(path):
    html_files = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith('.html'):
                html_files.append(os.path.join(root, file))
    return html_files

def get_html_file_name(file_path):
    # Get the file name without the directory and extension
    file_name = os.path.basename(file_path)
    file_name = os.path.splitext(file_name)[0]
    return file_name


all_dom_trees = get_html_files(path_to_dom_trees)

def generate_lighthouse_report(file_path, result_path=None):
    file_name = get_html_file_name(file_path)
    print(f"Generating Lighthouse report for {file_name}")

    url = f"http://localhost:{port}/{file_name}.html"

    if result_path is None:
        result_path = f"./../dataset/lh-original-reports/{file_name}.json"
    try:
        # Run Lighthouse and get the scores
        scores = run_lighthouse(url, result_path)
        if scores:
            print(f"Scores for {file_name}: {scores}")
        else:
            print(f"Failed to generate report for {file_name}")
    except Exception as e:
        logging.error(f"{file_name} --- An error occurred: {e}")
        print(f"An error occurred while generating the report for {file_name}: {e}")

def serve_directory(path, port=8080):
    # Change the working directory to the path with HTML files
    os.chdir(path)

    handler = http.server.SimpleHTTPRequestHandler
    httpd = socketserver.TCPServer(("", port), handler)

    # Run the server in a separate thread
    thread = threading.Thread(target=httpd.serve_forever)
    thread.daemon = True
    thread.start()
    return httpd, thread

def wait_until_available(url, timeout=10):
    for _ in range(timeout):
        try:
            r = requests.get(url)
            if r.status_code == 200:
                return True
        except requests.exceptions.RequestException as e:
            logging.error(f"Error connecting to {url}: {e}")
            return False
        time.sleep(1)
    return False

models = [
    # "claude-3-7-sonnet-20250219",
    # "claude-3-7-sonnet-20250219-non-reasoning",
    # "deepseek-r1",
    # "deepseek-v3-0324",
    # "gpt-4.1",
    "llama3.3-70b",
    # "o4-mini",
    # "qwen2.5-32b-instruct",
]

for model in models:
    try:
        path_to_dom_trees = f"./../results/reassembled-clean/{model}/"

        # server, thread = serve_directory(path_to_dom_trees, port=8080)
        all_dom_trees = get_html_files(path_to_dom_trees)

        # if not wait_until_available("http://localhost:8080"):
        #     print("Server did not start in time.")
        #     server.shutdown()
        #     continue

        for file_path in all_dom_trees:
            filename = get_html_file_name(file_path)
            
            result_path = f"./../dataset/lh-modified-reports/{model}/{filename}.json"
            if not os.path.exists(result_path):
                os.makedirs(os.path.dirname(result_path), exist_ok=True)

            generate_lighthouse_report(file_path, result_path)

        # server.shutdown()
        # thread.join()

    except Exception as e:
        logging.error(f"An error occurred while processing model {model}: {e}")
        print(f"An error occurred while processing model {model}: {e}")

# for file_path in all_dom_trees:
#     filename = get_html_file_name(file_path)
    
#     result_path = f"./../dataset/lh-original-reports/{filename}.json"

#     generate_lighthouse_report(file_path, result_path)

