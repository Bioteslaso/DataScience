function out = gaussfft(pic, t)
    %params:
        %pic - image matrix
        %t - variance
    %using the fast Fourier transform, 
    %convolves the image pic with a two-dimensional 
    %Gaussian function of arbitrary variance t via a discretization 
    %of the Gaussian function in the spatial domain
    %4steps
    
    %1. Generate a filter based on a sampled version of the Gaussian function
    %gaussian kernel - start with sampling points
    [xdim, ydim] = size(pic);
    [x, y] = meshgrid(-xdim/2 : (xdim/2)-1, -ydim/2 : (ydim/2)-1);
    
    gaussian_kernel = 1/(2*pi*t)*exp(-(x.^2+y.^2)./(2*t)); %the filter
    
    %2. Fourier transform the original image and the Gaussian filter.
    pic_hat = fft2(pic);
    gaussian_hat = fft2(gaussian_kernel);
    
    %3. Multiply the Fourier transforms.
    convolved_image = (pic_hat .* gaussian_hat);
    
    %4. Invert the resulting Fourier transform.
    inverted_pic = ifft2(convolved_image);
    out = fftshift(inverted_pic);
    
    %if we want to plot the freq domain instead
    %out = multiplication;
    
end
    