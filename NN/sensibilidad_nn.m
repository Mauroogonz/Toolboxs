function [p, indice] = sensibilidad_nn(Xent,net)
    [exported_ann_structure] = my_ann_exporter(net);

    % Parámetros para desnormalizar la salida entregada por la red
    y_max = exported_ann_structure.input_ymax;
    y_min = exported_ann_structure.input_ymin;
    x_max = exported_ann_structure.input_xmax;
    x_min = exported_ann_structure.input_xmin;

    % Regresores normalizados
    % Xentnor = (Xent-min(Xent))* (1--1)./(max(Xent)-min(Xent)) +-1;
    Xentnor = (Xent'-x_min)*(y_max-y_min)./(x_max-x_min) + y_min;
    % Salida normalizada
    %ynor = net.b{2} + net.LW{2}*tanh(net.b{1}+net.IW{1}*Xentnor);
    % Salida
    %y_nn = (ynor-y_min).*(x_max-x_min)./(y_max-y_min) + (x_min);

    diff_ynor = net.IW{1}'.*net.LW{2} * (1 - tanh(net.b{1}+net.IW{1}*Xentnor)).^2;
    indice = NaN(1,size(Xent,2));
    for i = 1:size(Xent,2)
        indice(i) = mean(diff_ynor(:,i)).^2 + std(diff_ynor(:,i)).^2;
    end
    
    efe=find(indice==min(indice));
    p = efe(1,1);
    %se grafican los indices obtenidos para compararse
    figure ()
    bar(indice)
    ylabel('Índice de sensibilidad')
    xlabel('Entrada del modelo')      
end