function [Position] = Code_position(anchor,Z)
% Calculate the actual position based on the distance measurement results 
% Use a simple least squares collaborative positioning method
% Input:anchor(Base station coordinate matrix),Z(Distance measurement matrix)
% Output:Position(Position coordinate matrix)

AP = anchor';
rtlength = Z';
[m,n] = size(Z);
xyz=zeros(3,n);      
for h=1:n        
    if length(AP)>=4
        A=[];b=[];      
        k = 1;
        for i1 = 1:m
             if rtlength(h,i1) ~= 0
                 L(k) = rtlength(h,i1);
                 AP_1(k,:) = AP(i1,:);
                 k = k+1;
             end
        end
        num_ap = length(AP_1);
        Q =zeros(1,num_ap);
        for j=1:num_ap
            Q(1,j) =0.5;
        end
        Q = diag(Q);      
        for i=1:num_ap     
            Node(i)=AP_1(i,1)^2+AP_1(i,2)^2+AP_1(i,3)^2;   
        end
        for i=1:num_ap   
            A=[A;2*AP_1(i,1),2*AP_1(i,2),2*AP_1(i,3)];
            b=[b;L(1)^2-L(i)^2+Node(i)-Node(1)];
        end
        x=inv(A'*inv(Q)*A)*(A'*inv(Q)*b);   
        xyz(:,h)=x; 
     end
end
Position = xyz'; 
end