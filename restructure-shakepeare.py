import re



file_read = open("shakespeare-1.txt", "r", encoding="utf-8")

file_write = open("Shakespeare/work-1.txt", "w", encoding="utf-8")

try:
	while True:
		line = file.readline()
