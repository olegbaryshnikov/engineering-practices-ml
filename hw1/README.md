## Homework 1  
1) install pipx (for poetry)  
```
python3 -m pip install --user pipx
python3 -m pipx ensurepath
```
2) install poetry  
```
pipx install poetry
```
3) create poetry project  
```
poetry new <path-to-project>
cd <path-to-project>
```
4) activate virtualenv (optional)  
```
poetry shell
```
5) install dev and prod dependencies  
```
poetry add -G dev black flake8 isort <other-dev-dependencies>
poetry add -G prod numpy <other-prod-dependencies>
```
6) use black formatter  
```
poetry run black --diff ./hw1/test.py # check diffs
poetry run black ./hw1/test.py # format file
```
7) use isort formatter
```
poetry run isort --check-only ./hw1/test.py # check is file formatted
poetry run isort ./hw1/test.py # format file
```
8) use flake8 linter
```
poetry run flake8 ./hw1/test.py
```