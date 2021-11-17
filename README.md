# text-generator
A RNN that can learn to write nonsense text. Put a bunch of .txt files in a folder named Data, start the main function and follow the instructions. Provided the computer has cuda functionality, the program will then do its best to learn the .txt files.

## Learning
The RNN learns the data character-wise. Meaning it takes a character and its "memory"-state as input and gives a character and a new "memory"-state as output. Meaning all it really learns is in what order characters appear in a text. 
