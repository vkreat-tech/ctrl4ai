pip install -r "requirements.txt" --user
if exist ctrl4ai.egg-info rmdir /s /q ctrl4ai.egg-info
python -m build
pip install --force-reinstall dist\ctrl4ai-1.0.24-py3-none-any.whl
