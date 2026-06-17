function [ res ] = my_ann_evaluation(net, input)
% Works with only single INPUT vector
% Matrix version can be implemented
ymax = net.input_ymax;
ymin = net.input_ymin;
xmax = net.input_xmax;
xmin = net.input_xmin;
input_preprocessed = (ymax-ymin) * (input-xmin) ./ (xmax-xmin) + ymin;
% Pass it through the ANN matrix multiplication
y1 = tanh(net.IW * input_preprocessed + net.b1);
y2 = net.LW * y1 + net.b2;
ymax = net.output_ymax;
ymin = net.output_ymin;
xmax = net.output_xmax;
xmin = net.output_xmin;
res = (y2-ymin) .* (xmax-xmin) /(ymax-ymin) + xmin;
end