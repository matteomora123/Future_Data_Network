function [data_part_TAG] = Code_recode(data_hex)
%Parse the stored data frames according to the communication protocol to obtain decimal ranging data
% Input:data_hex(HEX)
% Output:data_part_TAG(DEC)

i = 1;
k = 1;
data_s = hex2dec(data_hex);
data = data_s;

if(data(i)==85 && data(i+895)==238)
    p = 1;
    for q = i:i+660
        data_complete(k,p:p+1) = [data_hex(q,1),data_hex(q,2)];
        p = p+2;
    end
    i = i+896;
    k = k+1;
else
    i = i+1;
end

[N,] = size(data_complete);
for j = 1:N
    data_part(1,1:32,j) = data_complete(j,27:58);
    data_part_TAG(1,1,j) = hex2dec(strcat(data_part(1,3,j),data_part(1,4,j),data_part(1,1,j),data_part(1,2,j)));%Distance from UAV1 to Base_station1
    data_part_TAG(2,1,j) = hex2dec(strcat(data_part(1,7,j),data_part(1,8,j),data_part(1,5,j),data_part(1,6,j)));%Distance from UAV1 to Base_station2
    data_part_TAG(3,1,j) = hex2dec(strcat(data_part(1,11,j),data_part(1,12,j),data_part(1,9,j),data_part(1,10,j)));%Distance from UAV1 to Base_station3
    data_part_TAG(4,1,j) = hex2dec(strcat(data_part(1,15,j),data_part(1,16,j),data_part(1,13,j),data_part(1,14,j)));%Distance from UAV1 to Base_station4
    data_part_TAG(5,1,j) = hex2dec(strcat(data_part(1,19,j),data_part(1,20,j),data_part(1,17,j),data_part(1,18,j)));%Distance from UAV1 to Base_station5
    data_part_TAG(6,1,j) = hex2dec(strcat(data_part(1,23,j),data_part(1,24,j),data_part(1,21,j),data_part(1,22,j)));%Distance from UAV1 to Base_station6
    data_part_TAG(7,1,j) = hex2dec(strcat(data_part(1,27,j),data_part(1,28,j),data_part(1,25,j),data_part(1,26,j)));%Distance from UAV1 to Base_station7
    data_part_TAG(8,1,j) = hex2dec(strcat(data_part(1,31,j),data_part(1,32,j),data_part(1,29,j),data_part(1,30,j)));%Distance from UAV1 to Base_station8
    data_part(2,1:32,j) = data_complete(j,81:112); 
    data_part_TAG(1,2,j) = hex2dec(strcat(data_part(2,3,j),data_part(2,4,j),data_part(2,1,j),data_part(2,2,j)));%Distance from UAV2 to Base_station1
    data_part_TAG(2,2,j) = hex2dec(strcat(data_part(2,7,j),data_part(2,8,j),data_part(2,5,j),data_part(2,6,j)));
    data_part_TAG(3,2,j) = hex2dec(strcat(data_part(2,11,j),data_part(2,12,j),data_part(2,9,j),data_part(2,10,j)));
    data_part_TAG(4,2,j) = hex2dec(strcat(data_part(2,15,j),data_part(2,16,j),data_part(2,13,j),data_part(2,14,j)));
    data_part_TAG(5,2,j) = hex2dec(strcat(data_part(2,19,j),data_part(2,20,j),data_part(2,17,j),data_part(2,18,j)));
    data_part_TAG(6,2,j) = hex2dec(strcat(data_part(2,23,j),data_part(2,24,j),data_part(2,21,j),data_part(2,22,j)));
    data_part_TAG(7,2,j) = hex2dec(strcat(data_part(2,27,j),data_part(2,28,j),data_part(2,25,j),data_part(2,26,j)));
    data_part_TAG(8,2,j) = hex2dec(strcat(data_part(2,31,j),data_part(2,32,j),data_part(2,29,j),data_part(2,30,j)));
    data_part(3,1:32,j) = data_complete(j,135:166);    
    data_part_TAG(1,3,j) = hex2dec(strcat(data_part(3,3,j),data_part(3,4,j),data_part(3,1,j),data_part(3,2,j)));%Distance from UAV3 to Base_station1
    data_part_TAG(2,3,j) = hex2dec(strcat(data_part(3,7,j),data_part(3,8,j),data_part(3,5,j),data_part(3,6,j)));
    data_part_TAG(3,3,j) = hex2dec(strcat(data_part(3,11,j),data_part(3,12,j),data_part(3,9,j),data_part(3,10,j)));
    data_part_TAG(4,3,j) = hex2dec(strcat(data_part(3,15,j),data_part(3,16,j),data_part(3,13,j),data_part(3,14,j)));
    data_part_TAG(5,3,j) = hex2dec(strcat(data_part(3,19,j),data_part(3,20,j),data_part(3,17,j),data_part(3,18,j)));
    data_part_TAG(6,3,j) = hex2dec(strcat(data_part(3,23,j),data_part(3,24,j),data_part(3,21,j),data_part(3,22,j)));
    data_part_TAG(7,3,j) = hex2dec(strcat(data_part(3,27,j),data_part(3,28,j),data_part(3,25,j),data_part(3,26,j)));
    data_part_TAG(8,3,j) = hex2dec(strcat(data_part(3,31,j),data_part(3,32,j),data_part(3,29,j),data_part(3,30,j)));
    data_part(4,1:32,j) = data_complete(j,189:220);
    data_part_TAG(1,4,j) = hex2dec(strcat(data_part(4,3,j),data_part(4,4,j),data_part(4,1,j),data_part(4,2,j)));%Distance from UAV4 to Base_station1
    data_part_TAG(2,4,j) = hex2dec(strcat(data_part(4,7,j),data_part(4,8,j),data_part(4,5,j),data_part(4,6,j)));
    data_part_TAG(3,4,j) = hex2dec(strcat(data_part(4,11,j),data_part(4,12,j),data_part(4,9,j),data_part(4,10,j)));
    data_part_TAG(4,4,j) = hex2dec(strcat(data_part(4,15,j),data_part(4,16,j),data_part(4,13,j),data_part(4,14,j)));
    data_part_TAG(5,4,j) = hex2dec(strcat(data_part(4,19,j),data_part(4,20,j),data_part(4,17,j),data_part(4,18,j)));
    data_part_TAG(6,4,j) = hex2dec(strcat(data_part(4,23,j),data_part(4,24,j),data_part(4,21,j),data_part(4,22,j)));
    data_part_TAG(7,4,j) = hex2dec(strcat(data_part(4,27,j),data_part(4,28,j),data_part(4,25,j),data_part(4,26,j)));
    data_part_TAG(8,4,j) = hex2dec(strcat(data_part(4,31,j),data_part(4,32,j),data_part(4,29,j),data_part(4,30,j)));
    data_part(5,1:32,j) = data_complete(j,243:274);
    data_part_TAG(1,5,j) = hex2dec(strcat(data_part(5,3,j),data_part(5,4,j),data_part(5,1,j),data_part(5,2,j)));%Distance from UAV5 to Base_station1
    data_part_TAG(2,5,j) = hex2dec(strcat(data_part(5,7,j),data_part(5,8,j),data_part(5,5,j),data_part(5,6,j)));
    data_part_TAG(3,5,j) = hex2dec(strcat(data_part(5,11,j),data_part(5,12,j),data_part(5,9,j),data_part(5,10,j)));
    data_part_TAG(4,5,j) = hex2dec(strcat(data_part(5,15,j),data_part(5,16,j),data_part(5,13,j),data_part(5,14,j)));
    data_part_TAG(5,5,j) = hex2dec(strcat(data_part(5,19,j),data_part(5,20,j),data_part(5,17,j),data_part(5,18,j)));
    data_part_TAG(6,5,j) = hex2dec(strcat(data_part(5,23,j),data_part(5,24,j),data_part(5,21,j),data_part(5,22,j)));
    data_part_TAG(7,5,j) = hex2dec(strcat(data_part(5,27,j),data_part(5,28,j),data_part(5,25,j),data_part(5,26,j)));
    data_part_TAG(8,5,j) = hex2dec(strcat(data_part(5,31,j),data_part(5,32,j),data_part(5,29,j),data_part(5,30,j)));
    data_part(6,1:32,j) = data_complete(j,297:328);
    data_part_TAG(1,6,j) = hex2dec(strcat(data_part(6,3,j),data_part(6,4,j),data_part(6,1,j),data_part(6,2,j)));%Distance from UAV6 to Base_station1
    data_part_TAG(2,6,j) = hex2dec(strcat(data_part(6,7,j),data_part(6,8,j),data_part(6,5,j),data_part(6,6,j)));
    data_part_TAG(3,6,j) = hex2dec(strcat(data_part(6,11,j),data_part(6,12,j),data_part(6,9,j),data_part(6,10,j)));
    data_part_TAG(4,6,j) = hex2dec(strcat(data_part(6,15,j),data_part(6,16,j),data_part(6,13,j),data_part(6,14,j)));
    data_part_TAG(5,6,j) = hex2dec(strcat(data_part(6,19,j),data_part(6,20,j),data_part(6,17,j),data_part(6,18,j)));
    data_part_TAG(6,6,j) = hex2dec(strcat(data_part(6,23,j),data_part(6,24,j),data_part(6,21,j),data_part(6,22,j)));
    data_part_TAG(7,6,j) = hex2dec(strcat(data_part(6,27,j),data_part(6,28,j),data_part(6,25,j),data_part(6,26,j)));
    data_part_TAG(8,6,j) = hex2dec(strcat(data_part(6,31,j),data_part(6,32,j),data_part(6,29,j),data_part(6,30,j)));    
    data_part(7,1:32,j) = data_complete(j,351:382);
    data_part_TAG(1,7,j) = hex2dec(strcat(data_part(7,3,j),data_part(7,4,j),data_part(7,1,j),data_part(7,2,j)));%Distance from UAV7 to Base_station1
    data_part_TAG(2,7,j) = hex2dec(strcat(data_part(7,7,j),data_part(7,8,j),data_part(7,5,j),data_part(7,6,j)));
    data_part_TAG(3,7,j) = hex2dec(strcat(data_part(7,11,j),data_part(7,12,j),data_part(7,9,j),data_part(7,10,j)));
    data_part_TAG(4,7,j) = hex2dec(strcat(data_part(7,15,j),data_part(7,16,j),data_part(7,13,j),data_part(7,14,j)));
    data_part_TAG(5,7,j) = hex2dec(strcat(data_part(7,19,j),data_part(7,20,j),data_part(7,17,j),data_part(7,18,j)));
    data_part_TAG(6,7,j) = hex2dec(strcat(data_part(7,23,j),data_part(7,24,j),data_part(7,21,j),data_part(7,22,j)));
    data_part_TAG(7,7,j) = hex2dec(strcat(data_part(7,27,j),data_part(7,28,j),data_part(7,25,j),data_part(7,26,j)));
    data_part_TAG(8,7,j) = hex2dec(strcat(data_part(7,31,j),data_part(7,32,j),data_part(7,29,j),data_part(7,30,j)));    
    
end 

end