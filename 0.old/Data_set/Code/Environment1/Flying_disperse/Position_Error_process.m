function [Mse] = Position_Error_process(pri_positions,Position_true)
% Calculate the three-dimensional comprehensive positioning error for position calculation
% Input:pri_positions(The position coordinate matrix obtained from settlement)
% Input:Position_true(Predefine the position coordinate matrix of trajectory points)
% Output:Mse(three-dimensional comprehensive positioning error)

[m,~] = size(pri_positions);
Mse = zeros(m,1);
for i = 1:m
    Mse(i,1) = sqrt(((pri_positions(i,1)-Position_true(i,1))^2 + (pri_positions(i,2)-Position_true(i,2))^2 + (pri_positions(i,3)-Position_true(i,3))^2)/3);
end

end