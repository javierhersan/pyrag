# PyRag

Welcome to **PyRag**, a python RAG implementation

## PyRag setup.

Create python virtual environment:

```console
$ py -m venv .venv
```

Activate python virtual environment:

```console
$ .venv\scripts\activate
```

Update pip package installer

```console
$ pip3 install --upgrade pip
```

Add the following requirements to "requirements.txt" file:

- jupyter
- playwright

Install the requirements:

```console
$ python -m pip install -r requirements.txt
```

Add a dotenv (**_.env_**) file.

Add Open AI API key to the dotenv file.

```console
$ OPENAI_API_KEY="..."
```

Run the PyRag chat:

```console
$ py pyrag.py
```

## PyRag execution.

The following screenshot is a PyRag execution example.

![sample.png](./img/sample1.png)
![sample.png](./img/sample2.png)

## PyRag with jupyter notebooks.

Launch jupyter server on port 9999:

```console
$ jupyter notebook --port 9999
```
