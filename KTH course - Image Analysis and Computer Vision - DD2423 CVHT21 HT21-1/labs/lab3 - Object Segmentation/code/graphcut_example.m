scale_factor = 0.5;          % image downscale factor

% image region to train foreground with
area = [ 80, 110, 570, 300 ]  %tiger1
%area = [120, 160, 220, 170]  %tiger2
K = 20;                      % number of mixture components
alpha = 17.0;                 % maximum edge cost
sigma = 10.0;                % edge cost decay factor

I = imread('tiger3.jpg');
I_resized = imresize(I, scale_factor);
Iback = I_resized;
area_scaled = int16(area*scale_factor); 
[segm, prior] = graphcut_segm(I_resized, area_scaled, K, alpha, sigma);

Inew = mean_segments(Iback, segm);
I = overlay_bounds(Iback, segm);
imwrite(Inew,'results/gcut1.png')
imwrite(I,'results/gcut2.png')
imwrite(prior,'results/gcut3.png')
subplot(2,2,1); imshow(Inew); title('Final segmentation');
subplot(2,2,2); imshow(I); title('Overlaying bounds');
subplot(2,2,3); imshow(prior); title('Prior foreground probabilities');
