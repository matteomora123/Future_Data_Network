function [Dis_label_label] = Distance_L_L_comput(pri_positions)
%   Calculate the distance between the node to be located and the node to be located

[m,~] = size(pri_positions);
Dis_label_label = zeros(m,m);
for i = 1:m
    for j = 1:m
        Dis_label_label(i,j) = sqrt((pri_positions(i,1)-pri_positions(j,1))^2 + (pri_positions(i,2)-pri_positions(j,2))^2 + (pri_positions(i,3)-pri_positions(j,3))^2);
    end
end

end