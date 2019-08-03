import re

'''i = 1

f = open('data.tsv',"r",encoding="utf-8")
fw = open('corpus.tsv',"w",encoding="utf-8")
for line in f:
	tokens = line.strip().lower().split("\t")
	query_id,query,passage,label = tokens[0],tokens[1],tokens[2],tokens[3]
	#fw.write(passage+"\n")
	if i % 2455 == 0:
		break
	i = i + 1
		
f.close()
fw.close()
print(tokens)'''

def embedIt(embedFile):
	file = open(embedFile,'r',encoding='utf-8',errors='ignore')
	for line in file:
		tokens = line.strip().split()
		word = tokens[0]
		vec = tokens[1:]
		vec = " ".join(vec)
		gloveEmbeds[word] = vec
	gloveEmbeds['zerovec'] = '0.0 ' * embedDim
	file.close()


def convertIntoEmbed(inputFile, outputFile):
	#tokenise and convert words in input file to word vectors and load into output file
	inFile = open(inputFile,'r',encoding='utf-8',errors='ignore')
	outFile = open(outputFile,'w',encoding='utf-8')

	i = 0
	for line in inFile:
		tokens = line.strip().lower().split('\t')
		queryId, query, passage, passageId = tokens[0], tokens[1], tokens[2], tokens[3]
		#Convert queries and passages to vecs.. (download glove6b and extract to glove6b.txt)
		words = re.split('\W',query)
		words = [notEmptyWord for notEmptyWord in words if notEmptyWord]
		queryVec  = ''
		for word in words:
			if word in gloveEmbeds:
				queryVec += gloveEmbeds[word] + ' '
			else:
				queryVec += gloveEmbeds['zerovec']	+ ' '

		words = re.split('\W',passage)
		words = [notEmptyWord for notEmptyWord in words if notEmptyWord]
		passageVec = ''	
		for word in words:
			if word in gloveEmbeds:
				passageVec += gloveEmbeds[word] + ' '
			else:
				passageVec += gloveEmbeds['zerovec']	+ ' '

		outFile.write(queryId + '\t' + queryVec + '\t' + passageVec + '\t' + passageId + '\n')
		#print(queryVec + '\t' + passageVec + '\t' + label + '\n')
		if (i + 1) % 10000 == 0:
			print('Example ' + str(i + 1) + ' done')
		i = i + 1
		
		#print(re.split('\W',query))
		#print(query.split(' '))
		#break


if __name__ == '__main__':
	embedFile = 'glove.6B.50d.txt'
	gloveEmbeds = {}
	embedDim = 50
	gloveEmbeds['zerovec'] = '0.0 ' * embedDim
	embedIt(embedFile)	#enter glove embed txt file'''
	convertIntoEmbed('eval2_unlabelled.tsv','eval2_unlabelled_embedded.tsv') #evaluate.tsv contains queryId, queryVec, passageVec, label, passageId