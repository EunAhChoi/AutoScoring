import os.path


checkDirs = ['/Users/hcy/Desktop/GP/data/words/', '/Users/hcy/Desktop/GP/data/words/a01/a01-000u/']
checkFiles = ['/Users/hcy/Desktop/GP/data/words.txt', '/Users/hcy/Desktop/GP/data/test.png', '/Users/hcy/Desktop/GP/data/words/a01/a01-000u/a01-000u-00-00.png']


for f in checkDirs:
	if os.path.isdir(f):
		print('[OK]', f)
	else:
		print('[ERR]', f)


for f in checkFiles:
	if os.path.isfile(f):
		print('[OK]', f)
	else:
		print('[ERR]', f)
