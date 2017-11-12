function [X]=triangmidpoints(matches,P1,P2)

n=size(matches,1);

[U,S,V]=svd(P1);
Q1=V(:,4)./V(4,4);
t1=Q1(1:3);
invM1=invmat3x3(P1(1:3,1:3));
[U,S,V]=svd(P2);
Q2=V(:,4)./V(4,4);
t2=Q2(1:3);
invM2=invmat3x3(P2(1:3,1:3));

X=zeros(3,n);
Id=eye(3);

for i=1:n
  Xinf1=invM1*[matches(i,1);matches(i,2);1];
  D1=Xinf1./sqrt(sum(Xinf1.^2));
  Xinf2=invM2*[matches(i,3);matches(i,4);1];
  D2=Xinf2./sqrt(sum(Xinf2.^2));
  X(:,i)=invmat3x3((Id-D1*D1')+(Id-D2*D2'))*(t1+t2-(t1'*D1)*D1-(t2'*D2)*D2);
end


function invM=invmat3x3(M)

a=M(1,1);b=M(1,2);c=M(1,3);
d=M(2,1);e=M(2,2);f=M(2,3);
g=M(3,1);h=M(3,2);k=M(3,3);

detM=a*(e*k-f*h)+b*(f*g-k*d)+c*(d*h-e*g);

invM=[(e*k-f*h) (c*h-b*k) (b*f-c*e);...
      (f*g-d*k) (a*k-c*g) (c*d-a*f);...
      (d*h-e*g) (g*b-a*h) (a*e-b*d)];
invM=invM./detM;

if 0
  max(max(abs(invM-inv(M))));
end
