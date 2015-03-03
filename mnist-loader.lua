require 'csvigo'
require 'torch'

--data format: each row represents an image, and each column (except for the first column) is a pixel, we have 785 columns. 
--First column represents image label and the remaining 784 are the pixels of the image.

-- uses csvigo to parse the csv.
-- skip records is used to skip the header, if present.

-- converts the table returned by csvigo into a torch Tensor (default Tensor)
local parser = function(csvData,skip_records, imageWidth, imageHeight) 
	numExamples = #csvData-skip_records
	print('num examples: '..numExamples)
	X = torch.Tensor(numExamples,imageWidth,imageHeight)
	Y = torch.Tensor(numExamples)

	for idx = 2,#csvData do
		data = torch.Tensor(csvData[idx])
		Y[idx-1] = data[1]
		X[idx-1] = data[{ {2,imageWidth*imageHeight+1} }]:reshape(imageWidth,imageHeight)
	end
	return X,Y
end

-- Parses the mnit data set in csv format.	
-- Returns two Tensors, X represents the images, Y represents their labels.
-- X is a 28x28 Tensor, Y is 1 dim tensor containing #image elements.
function loaddataset(options)
	csvData = csvigo.load{path=options.file_path,mode='raw'}
	return parser(csvData,options.skip_records,options.image_width,options.image_height)
end