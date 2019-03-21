import os

image_files = os.listdir("images")
print(len(image_files))
for image_file in image_files:
	print(image_file)
	image_class = image_file.split(".")[0]

	if not os.path.exists("images/" + image_class):
		os.mkdir("images/" + image_class)
	#print("images/"+image_file)
	#print("images/"+image_class+"/"+image_file)
	os.rename("images/"+image_file, "images/" + image_class + "/" +  image_file)
