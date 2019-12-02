import os 
from tqdm import tqdm

dir = "./Midi_Files/"
temp_dir = "./Temp/"

# Convert all .midi files into .abc files
for filename in os.listdir(dir):
    if filename != '.DS_Store':
        command = "midi2abc -f ./Midi_Files/" + filename + " -o ./Temp/ABC_" + filename
        os.system(command)

# Merge all .abc files into one single .abc file with double spaces seperating songs
file_write = open("../Data_Tunes.txt","w+")
for filename in os.listdir(temp_dir):
    if filename != '.DS_Store':
        file_read = open(os.path.join(temp_dir, filename), mode = 'r')
        data = file_read.read()
        file_read.close()
        file_write.write(data)
        file_write.write("\n") # Formatting
        file_write.write("\n")