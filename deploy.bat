pip install twine --user

set token="pypi-"

python -m twine upload -u __token__ -p %token% --repository pypi dist\ctrl4ai-1.0.24.tar.gz
python -m twine upload -u __token__ -p %token% --repository pypi dist\ctrl4ai-1.0.24-py3-none-any.whl
