function [ Rmean ] = dqq_L1_mean_rotation_matrix( R )
%DQQ_L1_MEAN_ROTATION_MATRIX Summary of this function goes here
%   This function calculate the mean rotation matrix of the given 3*3*n R matrix
%	under L1 norm by Weiszfeld algorithm.
%	Please refer to the paper: 
%	'L1 rotation averaging using the Weiszfel algorithm', Richard Hartley, etc, CVPR 2011
%	for details.

S(:,:,1) = dqq_rotation_quaternion_initialization( R );
nofR=size(R);

iter=1;

while isreal(S(:,:,iter))
    iter=iter+1;
    sum_vmatrix_normed(:,:,iter)=zeros(3,3);
    for j=1:nofR(3)
        vmatrix(:,:,j)=logm(R(:,:,j)*(S(:,:,iter-1))^(-1));
        vmatrix_normed(:,:,j)=vmatrix(:,:,j)/norm(vmatrix(:,:,j));
        sum_vmatrix_normed(:,:,iter)=sum_vmatrix_normed(:,:,iter)+vmatrix_normed(:,:,j);
        inv_norm_vmatrix(j)=1/norm(vmatrix(:,:,j));
    end
    
    delta(:,:,iter)=sum_vmatrix_normed(:,:,iter)/sum(inv_norm_vmatrix);
    
    S(:,:,iter)=expm(delta(:,:,iter))*S(:,:,iter-1);
    
    if abs(1-det(S(:,:,iter)*S(:,:,iter)'))<10^(-10)
        break;
    end
end

Rmean=S(:,:,iter-1);

end

