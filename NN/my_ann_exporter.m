function [ my_ann_structure ] = my_ann_exporter(trained_netw)
% Just for extracting as Structure object
my_ann_structure.input_ymax = trained_netw.inputs{1}.processSettings{1}.ymax;
my_ann_structure.input_ymin = trained_netw.inputs{1}.processSettings{1}.ymin;
my_ann_structure.input_xmax = trained_netw.inputs{1}.processSettings{1}.xmax;
my_ann_structure.input_xmin = trained_netw.inputs{1}.processSettings{1}.xmin;
my_ann_structure.IW = trained_netw.IW{1};
my_ann_structure.b1 = trained_netw.b{1};
my_ann_structure.LW = trained_netw.LW{2};
my_ann_structure.b2 = trained_netw.b{2};
my_ann_structure.output_ymax = trained_netw.outputs{2}.processSettings{1}.ymax;
my_ann_structure.output_ymin = trained_netw.outputs{2}.processSettings{1}.ymin;
my_ann_structure.output_xmax = trained_netw.outputs{2}.processSettings{1}.xmax;
my_ann_structure.output_xmin = trained_netw.outputs{2}.processSettings{1}.xmin;
end
