import subprocess

envs = ['COGS108_FA25', 'CSE151A', 'dsc80', 'math189']
for env in envs:
    res = subprocess.run(['conda', 'run', '-n', env, 'python', '-c', 'import pandas, lightgbm, sklearn; print("OK")'], capture_output=True, text=True)
    print(env, "exit code:", res.returncode, "stdout:", res.stdout.strip(), "stderr:", res.stderr.strip())
