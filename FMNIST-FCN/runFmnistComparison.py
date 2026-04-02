import multiprocessing
import subprocess

frac = 100
fract = float(frac)/100.0

def run_script(command):
    subprocess.run(command, check=True,  capture_output=True, text=True)

commands = []
for model in ['small_fcn', 'medium_fcn', 'large_fcn']:
    for fbAtStart in ['0']:
        for iteration in range(5):
            seed = str(iteration + 100)

            if model == 'large_fcn':
                epochs = '1000'
            else:
                epochs = '500'

            # #Clean
            command = []
            command.append('/bin/python3')
            command.append('FMNIST-FCN/FashionMnist.py')
            command.append('--model')
            command.append(model)
            command.append('--gpu')
            command.append('0')
            command.append('--epochs')
            command.append(epochs)
            command.append('--lr')
            command.append('.001')
            command.append('--seed')
            command.append(seed)
            command.append('--neural_balance')
            command.append('0')
            command.append('--neuralFullBalanceAtStart')
            command.append(fbAtStart)
            command.append('--filename')
            command.append(f'{model}-Clean-{iteration}-fbAtStart-{fbAtStart}-seed-{seed}-frac-{fract}')
            command.append(f'--foldername')
            command.append(f'{model}')
            commands.append(command)

            #L1
            command = []
            command.append('/bin/python3')
            command.append('FMNIST-FCN/FashionMnist.py')
            command.append('--model')
            command.append(model)
            command.append('--gpu')
            command.append('0')
            command.append('--epochs')
            command.append(epochs)
            command.append('--l2_weight')
            command.append('.015')
            command.append('--lr')
            command.append('.001')
            command.append('--seed')
            command.append(seed)
            command.append('--neural_balance')
            command.append('0')
            command.append('--neuralFullBalanceAtStart')
            command.append(fbAtStart)
            command.append('--order')
            command.append('1')
            command.append('--filename')
            command.append(f'{model}-L1Regularization-{iteration}-fbAtStart-{fbAtStart}-seed-{seed}-frac-{fract}')
            command.append(f'--foldername')
            command.append(f'{model}')
            commands.append(command)

            #L2
            command = []
            command.append('/bin/python3')
            command.append('FMNIST-FCN/FashionMnist.py')
            command.append('--model')
            command.append(model)
            command.append('--gpu')
            command.append('0')
            command.append('--epochs')
            command.append(epochs)
            command.append('--l2_weight')
            command.append('.015')
            command.append('--lr')
            command.append('.001')
            command.append('--seed')
            command.append(seed)
            command.append('--neural_balance')
            command.append('0')
            command.append('--neuralFullBalanceAtStart')
            command.append(fbAtStart)
            command.append('--order')
            command.append('1')
            command.append('--filename')
            command.append(f'{model}-L2Regularization-{iteration}-fbAtStart-{fbAtStart}-seed-{seed}-frac-{fract}')
            command.append(f'--foldername')
            command.append(f'{model}')
            commands.append(command)

for i in commands:
    for j in i:
        print(j, end=' ')
    print()

max_processes = min(8, multiprocessing.cpu_count())
with multiprocessing.Pool(processes=max_processes) as pool:
    print(max_processes)
    pool.map(run_script, commands)