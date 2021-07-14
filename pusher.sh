for i in $(exa demos/*.ipynb); do jupytext --to script "demos/$i"; done;
black "."
isort .
# pdoc --force --html -o docs sprintdl
# mv docs/sprintdl/index.html docs/index.md
# mv docs/sprintdl/* docs/
if [[ ! -z $1 ]]; then
        git add . && git commit -m $1 && git push
fi
