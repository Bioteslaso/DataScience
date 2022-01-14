function fftwave(u, v, sz)
% e.g.: fftwave(5, 9, 128)
    if (nargin < 2)
        error('Requires at least two input arguments.')
    end
    
    %If only two args - use default size 128
    if (nargin == 2)
        sz = 128;
    end
    
    %Else use provided size by param
    %we never compute any actual Fourier transf here
    %instead we directly input the values for our Fourier transf
    Fhat = zeros(sz);
    Fhat(u, v) = 1;
    
    F = ifft2(Fhat);
    Fabsmax = max(abs(F(:)));
    subplot(3, 2, 1);
    showgrey(Fhat);
    title(sprintf('Fhat: (u, v) = (%d, %d)', u, v))

    % What is done by these instructions?
    if (u <= sz/2)
        uc = u - 1; 
    else
        uc = u - 1 - sz;
    end
    if (v <= sz/2)
        vc = v - 1;
    else
        vc = v - 1 - sz;
    end
    
    
    %----- Q4 ----------------
    %WAVELENGTH lambda
    % Replace by correct expression
    w_1 = 2*pi*uc/sz;
    w_2 = 2*pi*vc/sz;
    wavelength = 2*pi/sqrt(w_1^2+w_2^2);
    %--------------------------
    
    %----- Q3 -----------------
    %AMPLITUDE A
    %Matlab - convention for scaling: 
        %ifft function in MATLAB includes a scaling factor of 1/M^2 
        %as part of the computation
    %i.e. the factor *1/sz^2 = *1/128^2 at the end
    %max(abs(Fhat(:))) = 1/sz
    %amplitude = max(abs(Fhat(:)))/sz^2;
    amplitude = max(abs(Fhat(:))/sz)
    %---------------------------
    
    subplot(3, 2, 2);
    %ffthshift - shift zero-frequency component to center of spectrum.
    %showgrey(fftshift(Fhat));
    
    shiftedFhat = fftshift(Fhat);
    %Add point in new center
    %shiftedFhat(sz/2+1,sz/2+1) = 1;
    showgrey(shiftedFhat);
  
    title(sprintf('centered Fhat: (uc, vc) = (%d, %d)', uc, vc))
    
    subplot(3, 2, 3);
    showgrey(real(F), 64, -Fabsmax, Fabsmax);
    title('real(F)')
    
    subplot(3, 2, 4);
    showgrey(imag(F), 64, -Fabsmax, Fabsmax);
    title('imag(F)')
    
    subplot(3, 2, 5);
    showgrey(abs(F), 64, -Fabsmax, Fabsmax);
    title(sprintf('abs(F) (amplitude %f)', amplitude))
    
    subplot(3, 2, 6);
    showgrey(angle(F), 64, -pi, pi);
    title(sprintf('angle(F) (wavelength %f)', wavelength))
    
    %plot 3D of imaginary part of the spatial domain
    figure(2);
    surf(imag(F));
    colormap winter;
    title('imag(F) - 3D visualization')
    