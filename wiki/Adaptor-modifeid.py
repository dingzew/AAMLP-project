import random
import csv

random.seed(15)

# save the data from source file
dic = {}
question2id = {}

def readSomeQuestions(n = 100, file = "part-00000"):
    """
    Read some lines of data into dictionary
    :param n: How many line read from source file
    :param file: The file path of the source file
    :return: Not applicable
    """
    questions = []
    with open(file, "r") as input:
        for i in range(n):
            dic[i] = input.readline()
            dic[i] = dic[i].replace("\n", "")
            dic[i] = dic[i].split("\t")
            for j in range(len(dic[i])):
                dic[i][j] = dic[i][j][2:]
                questions.append(dic[i][j])

    # Generate shuffled question ids
    random.shuffle(questions)
    for i, q in enumerate(questions):
        question2id[q] = i


def generateSimilarQuestion(n = 20):
    """
    Generate n duplicated questions
    :param n: How many duplicated questions to generate
    :return: N Question pairs
    """
    res = []
    size = len(dic)
    for i in range(n):
        line = random.randint(0, size - 1)
        lineSize = len(dic[line])
        q1 = random.randint(0, lineSize - 1)
        q2 = random.randint(0, lineSize - 1)
        if lineSize > 1:
            while q2 == q1:
                q2 = random.randint(0, lineSize - 1)
        res.append([dic[line][q1], dic[line][q2]])
    return res


def generateDifferentQuestion(n = 20):
    """
    Generate n non-duplicated questions
    :param n: How many non-duplicated questions to generate
    :return: N Question pairs
    """
    res = []
    size = len(dic)
    for i in range(n):
        line1 = random.randint(0, size - 1)
        line2 = random.randint(0, size - 1)
        while line2 == line1:
            line2 = random.randint(0, size - 1)
        index1 = random.randint(0, len(dic[line1]) - 1)
        index2 = random.randint(0, len(dic[line2]) - 1)
        res.append([dic[line1][index1], dic[line2][index2]])
    return res


def writeToCSV(questionPairs = [], similarNum = 0, file = "test_output.csv"):
    """
    Give the question pairs
    :param questionPairs: All the question pairs
    :param similarNum: How many duplicated questions to generate
    :param file: The output file
    :return: Not applicable
    """
    with open(file, "w") as output:
        toWrite = []
        writer = csv.writer(output, delimiter=',',
                            quotechar='"', quoting=csv.QUOTE_MINIMAL)
        firstLine = ["id","qid1","qid2","question1","question2","is_duplicate"]
        writer.writerow(firstLine)
        size = len(questionPairs)
        for i in range(size):
            outputLine = []
            outputLine.append(questionPairs[i][0])
            outputLine.append(questionPairs[i][1])
            if i < similarNum:
                outputLine.append(1)
            else:
                outputLine.append(0)
            toWrite.append(outputLine)
        random.shuffle(toWrite)
        for i in range(len(toWrite)):
            # Use shuffled question ids
            q1_id, q2_id = question2id[toWrite[i][0]], question2id[toWrite[i][1]]
            toWrite[i] = [i, q1_id, q2_id] + toWrite[i]
            writer.writerow(toWrite[i])


def pipeline(datasetNum = 20000, objectLength = 10000, similarQuestionRate = 0.5,
             inputPath = "part-00000", outputPath = "test_output.csv"):
    """
    Whole pipeline of the program
    :param datasetNum: How many lines read from source file
    :param objectLength: How many lines write to the output file
    :param similarQuestionRate: What is the percentage of duplicated question
    :param inputPath: Input file path
    :param outputPath: Output file path
    :return: Not applicable
    """
    readSomeQuestions(datasetNum, inputPath)
    similarQuestionNum = int(objectLength * similarQuestionRate)
    differentQuestionNum = objectLength - similarQuestionNum
    similarQuestions = generateSimilarQuestion(similarQuestionNum)
    differentQuestions = generateDifferentQuestion(differentQuestionNum)
    res = similarQuestions + differentQuestions
    writeToCSV(res, similarQuestionNum, outputPath)


def main():
    pipeline(datasetNum = 400000, objectLength = 1600000, similarQuestionRate = 0.3,
             inputPath = "part-00000", outputPath = "test_output.csv")


if __name__ == "__main__":
    main()