black "."
isort .
# pdoc --force --html -o docs sprintdl
# mv docs/sprintdl/index.html docs/index.md
# mv docs/sprintdl/* docs/
# rm demos/*.py
for i in $(exa demos); do jupytext --to script "demos/$i"; done;
if [[ ! -z $1 ]]; then
        git add . && git commit -m $1 && git push
fi
