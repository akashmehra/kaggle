local has_module, module = pcall(require,'csvigo')
csvigo_installed = has_module and module or nil
if csvigo_installed == nil then
	print('csvigo not found...Installing\nExecuting luarocks install csvigo')
	os.execute('luarocks install csvigo')
	require 'csvigo'
end
require 'torch'

--data format: each row represents an image, and each column (except for the first column) is a pixel,
--we have 785 columns. 
--First column represents image label and the remaining 784 are the pixels of the image.
 
-- uses csvigo to parse the csv.
-- skip records is used to skip the header, if present.
 
-- converts the table returned by csvigo into a torch Tensor (default Tensor)
local parser = function(csvData,maxLoad,skipRecords,geometry)
	numExamples = math.min(#csvData,maxLoad)
	local imageWidth = geometry[1]
	local imageHeight = geometry[2]
	X = torch.Tensor(numExamples,imageWidth,imageHeight)
	Y = torch.Tensor(numExamples)
 
	for idx = 1+skipRecords,numExamples+skipRecords do
		data = torch.Tensor(csvData[idx])
		Y[idx-1] = data[1]
		X[idx-1] = data[{ {2,imageWidth*imageHeight+1} }]:reshape(imageWidth,imageHeight)
	end
	return X,Y
end

local loaddataset = function(filePath,maxLoad,skipRecords,geometry)
	csvData = csvigo.load{path=filePath,mode='raw'}
	return parser(csvData,maxLoad,skipRecords,geometry)
end

mnist = {}

function mnist.loadTrainSet(pathTrainSet,maxLoad,skipRecords,geometry)
   	return mnist.loadDataset(pathTrainSet,maxLoad,skipRecords,geometry)
end

function mnist.loadTestSet(pathTestSet,maxLoad,skipRecords,geometry)
   	return mnist.loadDataset(pathTestSet,maxLoad,skipRecords,geometry)
end


-- Parses the mnit data set in csv format.	
-- Returns two Tensors, X represents the images, Y represents their labels.
-- X is a 28x28 Tensor, Y is 1 dim tensor containing #image elements.
-- just call this function lie this:
-- dataset = mnist.loadTrainSet('../train.csv',42000,1,{28,28})
-- should return a table containing the following:
-- {
--  data : DoubleTensor - size: 42000x28x28
--  labels : DoubleTensor - size: 42000
--  size : function: 0x23803d00
--  normalizeGlobal : function: 0x23d26c78
--  normalize : function: 0x38b39d80
-- }
function mnist.loadDataset(fileName,maxLoad,skipRecords,geometry)

   	data, labels = loaddataset(fileName,maxLoad,skipRecords,geometry)
   	local dataset = {}
   	dataset.data = data
   	dataset.labels = labels

   	function dataset:normalize(mean_, std_)
    	local mean = mean or data:view(data:size(1), -1):mean(1)
     	local std = std_ or data:view(data:size(1), -1):std(1, true)
     	for i=1,data:size(1) do
        	data[i]:add(-mean[1][i])
        	if std[1][i] > 0 then
           		tensor:select(2, i):mul(1/std[1][i])
         	end
      	end
    	return mean, std
  	end

   	function dataset:normalizeGlobal(mean_, std_)
   		local std = std_ or data:std()
    	local mean = mean_ or data:mean()
    	data:add(-mean)
    	data:mul(1/std)
    	return mean, std
   	end

   	function dataset:size()
		return nExample
   	end

   	local labelvector = torch.zeros(10)

   	setmetatable(dataset, {__index = function(self, index)
					local input = self.data[index]
					local class = self.labels[index]
					local label = labelvector:zero()
					label[class] = 1
					local example = {input, label}
					return example
   	end})

	return dataset
end

return mnist