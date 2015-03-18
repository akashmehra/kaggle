require 'torch'

cross_validate = {}

-- splits the dataset into train and validation sets based on the splitSize (a float) passed in as an argument.
-- the function signature is [validationSet,trainSet] splitDataSet (input, splitSize,shuffle)
-- input: Input Tensor (should have atleast one dimension
-- splitSize: a floating point value ranging between [0,1], for instance 0.3 (technically inclusive
-- but doesn't make sense to call this function with the interval endpoints).
-- shuffle: a boolean, if true then the dataset is shuffled and split, if the value is false then the dataset only split.
-- Note: The input Tensor remains the same even if shuffling is turned on.

function cross_validate.splitDataSet(input,splitSize,shuffle)
    if input ~= nil and input:nDimension() >= 1 then
        if shuffle == nil then
            shuffle = true
        end
        if shuffle == true then
            indices = torch.randperm(input:size(1)):long()
        else
            indices = torch.range(1,input:size(1)):long()
        end
        
        intPart,fracPart = math.modf(splitSize * input:size(1))
        if fracPart >= 0.5 then
            intPart = intPart + 1
        end
        
        validationSet = input:index(1,indices[{ {1,intPart} }])
        trainSet = input:index(1, indices[{ {intPart+1,input:size(1)} }])
        return validationSet, trainSet
    end
    return nil
end
