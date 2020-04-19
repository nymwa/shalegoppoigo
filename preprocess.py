from collections import Counter

if __name__ == '__main__':
	with open('dict.txt') as f:
		words = f.read().splitlines()
	chars = Counter([x for xs in words for x in xs]).most_common()
	
	vlist = ['[END]'] + [x for x, _ in chars]
	vdict = {x:i for i, x in enumerate(vlist)}

	with open('vocab.txt', 'w') as f:
		for x in vlist:
			print(x, file=f)

	with open('train.txt', 'w') as f:
		for x in words:
			print(' '.join(list(x)), file=f)
