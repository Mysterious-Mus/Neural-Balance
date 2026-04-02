import multiprocessing
import subprocess
import random

lr = '.001'
epochs = '250'
frac = 5
fract = float(frac)/100.0
def run_script(command):
    subprocess.run(command, check=True)

commands = []

for model in ['3']:
    for fbAtStart in ['0']:
        for iteration in range(5):
            # l2 reg
            command = []
            command.append('/bin/python3')
            command.append('IMDB-RNN/rnn_imdb.py')
            command.append('--n_layers')
            command.append(model)
            command.append('--gpu')
            command.append('0')
            command.append('--l2_weight')
            command.append('.0001')
            command.append('--lr')
            command.append(lr)
            command.append('--seed')
            command.append(str(iteration+100))
            command.append('--neural_balance')
            command.append('0')
            command.append('--neuralFullBalanceAtStart')
            command.append(fbAtStart)
            command.append('--foldername')
            command.append('imdb')
            command.append('--filename')
            command.append(f'L2Regularization-{iteration}-fbAtStart-{fbAtStart}-seed-{iteration+100}-frac-{frac}')
            command.append('--epochs')
            command.append(epochs)
            command.append('--trainDataFrac')
            command.append(str(fract))
            commands.append(command)

            #clean

            command = []
            command.append('/bin/python3')
            command.append('IMDB-RNN/rnn_imdb.py')
            command.append('--n_layers')
            command.append(model)
            command.append('--gpu')
            command.append('2')
            command.append('--lr')
            command.append(lr)
            command.append('--seed')
            command.append(str(iteration+100))
            command.append('--neural_balance')
            command.append('0')
            command.append('--foldername')
            command.append('imdb')
            command.append('--neuralFullBalanceAtStart')
            command.append(fbAtStart)
            command.append('--filename')
            command.append(f'Clean-{iteration}-fbAtStart-{fbAtStart}-seed-{iteration+100}-frac-{frac}')
            command.append('--epochs')
            command.append(epochs)
            command.append('--trainDataFrac')
            command.append(str(fract))
            commands.append(command)

            
            #L2 NB

            command = []
            command.append('/bin/python3')
            command.append('IMDB-RNN/rnn_imdb.py')
            command.append('--n_layers')
            command.append(model)
            command.append('--gpu')
            command.append('4')
            command.append('--lr')
            command.append(lr)
            command.append('--seed')
            command.append(str(iteration+100))
            command.append('--neural_balance')
            command.append('1')
            command.append('--order')
            command.append('2')
            command.append('--foldername')
            command.append('imdb')
            command.append('--neuralFullBalanceAtStart')
            command.append(fbAtStart)
            command.append('--filename')
            command.append(f'L2NB-{iteration}-{iteration}-fbAtStart-{fbAtStart}-seed-{iteration+100}-frac-{frac}')
            command.append('--epochs')
            command.append(epochs)
            command.append('--trainDataFrac')
            command.append(str(fract))
            commands.append(command)

# # Create a pool of workers and map commands to the workers
with multiprocessing.Pool(processes=3) as pool:
    pool.map(run_script, commands)