function [Anchor] = Get_Anchor(Flag)
% Obtain the reference base station position coordinates in the corresponding scenario
% Flag = 0:Environment0-outdoor;
% Flag = 1:Environment1-indoor;

if Flag == 0
    Anchor = load('I:\Data_set\Environment0\Anchors.mat');
else
    Anchor = load('I:\Data_set\Environment1\Anchors.mat');
end

Anchor = Anchor.Anchors;
end