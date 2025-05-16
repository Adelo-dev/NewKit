<h1 align="center">
  <br>
  <img src="https://i.imgur.com/6dmGfVb.png" alt="Newkit" width="200"></a>
  <br>
  NewKit
  <br>
</h1>

# Quick start guide:

Install "uv" based on your OS: 

[uv installation documentation](https://docs.astral.sh/uv/getting-started/installation/#__tabbed_1_2)

Clone Repository:

```
git clone https://github.com/Adelo-dev/NewKit
```

Run main.py through uv to create a venv:
```
uv run main.py
```

# Extra
Adding new packages:
```
uv add <package_name>
```

Use ruff to check for linter and code formatting issues:
```
uvx ruff check 
```
# How to create a dataset

To create a dataset for to detect movement you will two sets of csv files and video samples. 
Take sample videos of both the incorrect and correct form of the excerise can label them.
Split the videos into into up and down based on their highest and lowest point. Then split into frames.

--need to clarify where to put the frames and which functions to use --

The first csv file will contain the excerise movement split into two parts up and down
The second csv file will contain the errors and the correct movement pattern for the excerise