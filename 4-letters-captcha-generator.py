# coding=utf-8
from numpy import arange
from captcha.image import ImageCaptcha
from string import ascii_lowercase, ascii_uppercase
import random


def generate():
	lower = [letter for letter in ascii_lowercase] # all lowercase characters
	upper = [letter for letter in ascii_uppercase] # all uppercase characters
	numbers = [str(num) for num in arange(0,10)] # [0-9]

	charset = lower + upper + numbers

	image = ImageCaptcha(fonts=['/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf', '/usr/share/fonts/truetype/liberation/LiberationSans-Italic.ttf'])
	random.seed();
	
	captcha_for_char = int(raw_input("How many captcha for char do you want? "))

	for a in arange(0,len(charset)):
		for b in arange(0,captcha_for_char):
			word = "".join([charset[a], charset[random.randint(0, len(charset)-1)], charset[random.randint(0, len(charset)-1)], charset[random.randint(0, len(charset)-1)]])
			image.write(word, word+'.png')
			print("generated " + word)


def main():
	generate()
	return


if __name__ == '__main__':
	main()