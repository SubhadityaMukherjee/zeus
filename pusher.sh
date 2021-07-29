black "."
isort .
jupytext --to notebook demos/*.py
rm -rf demos/notebooks/*WIP*
mv demos/*.ipynb demos/notebooks/
# pdoc --force --html -o docs sprintdl
# mv docs/sprintdl/index.html docs/index.md
# mv docs/sprintdl/* docs/
git add . && git commit -m $1 && git push
