% Part 1.3 - Basis functions - Questions 1 to 6


sz = 128

points = [1 1; 65 65; 65 66; 66 65; 5 9; 9 5; 17 9; 17 121; 5 1; 125 1];
for i=1:length(points)
    fftwave(points(i,1), points(i,2), sz);
    pause();
end


%% Part 1.4 - Linearity - Questions 7 to 9


F = [zeros(56, 128); ones(16, 128); zeros(56, 128)];
G = F';
H = F + 2 * G;

Fhat = fft2(F);
Ghat = fft2(G);
Hhat = fft2(H);

subplot(3, 4, 1);
showgrey(F);
title("F");
subplot(3, 4, 2);
showgrey(G);
title("G (F^T)");
subplot(3, 4, 3);
showgrey(H);
title("H (F+2*G)");
subplot(3, 4, 4);
showgrey(log(1 + abs(Fhat)));
title("log(1 + abs(Fhat))");
subplot(3, 4, 5);
showgrey(1 + abs(fftshift(Fhat)));
title("1 + abs(fftshift(Fhat))");
subplot(3, 4, 6);
showgrey(log(1 + abs(Ghat)));
title("log(1 + abs(Ghat))");
subplot(3, 4, 7);
showgrey(log(1 + abs(Hhat)));
title("log(1 + abs(Hhat))");
subplot(3, 4, 8);
showgrey(log(1 + abs(fftshift(Hhat))));
title("log(1 + abs(fftshift(Hhat)))");
subplot(3, 4, 9);
showgrey(1 + abs(fftshift(Hhat)));
title("1 + abs(fftshift(Hhat))");

subplot(2, 3, 1);
showgrey(F);
title("F");
subplot(2, 3, 2);
showgrey(G);
title("G (F^T)");
subplot(2, 3, 3);
showgrey(H);
title("H (F+2*G)");
subplot(2, 3, 4);
showgrey(log(1 + abs(fftshift(Fhat))));
title("log(1 + abs(fftshift(Fhat)))");
subplot(2, 3, 5);
showgrey(log(1 + abs(fftshift(Ghat))));
title("log(1 + abs(fftshift(Ghat)))");
subplot(2, 3, 6);
showgrey(log(1 + abs(fftshift(Hhat))));
title("log(1 + abs(fftshift(Hhat)))");


F = [zeros(56, 128); ones(16, 128); zeros(56, 128)];
G = F';
H = F .* G;

subplot(1,2,1)
showgrey(F .* G);
title('F*G')
subplot(1,2,2)
showfs(fft2(F .* G));
title('FT{F*G}')


%% Part 1.5 - Multiplication - Question 10


F = [zeros(56, 128); ones(16, 128); zeros(56, 128)];
G = F';
H = F .* G;
Fhat = fft2(F);
Ghat = fft2(G);
Hhat = fft2(H);

subplot(2, 3, 1);
showgrey(F);
title("F");
subplot(2, 3, 2);
showgrey(G);
title("G");
subplot(2, 3, 3);
showgrey(H);
title("F .* G");
subplot(1, 2, 1);
showfs(fftshift(conv2(fftshift(Fhat)/128,fftshift(Ghat)/128, 'same')));
title("conv2(Fhat, Ghat)");
subplot(1, 2, 2);
showgrey(log(1 + abs(fftshift(Hhat))));
title("fft2(F .* G)");


%% Part 1.6 - Scaling - Question 11


prev = [zeros(56, 128); ones(16, 128); zeros(56, 128)];
prev = prev .* prev';
F = [zeros(60, 128); ones(8, 128); zeros(60, 128)] .* [zeros(128, 48) ones(128, 32) zeros(128, 48)];
subplot(2, 2, 1);
showgrey(prev);
title("F_{before}");
subplot(2, 2, 2);
showfs(fft2(prev));
title("fft2(F_{before})");
subplot(2, 2, 3);
showgrey(F);
title("F_{new}");
subplot(2, 2, 4);
showfs(fft2(F));
title("fft2(F_{new})");


%% Part 1.7 - Rotation - Question 12


F = [zeros(60, 128); ones(8, 128); zeros(60, 128)] .* [zeros(128, 48) ones(128, 32) zeros(128, 48)];
alphas =  [30 45 60 90];

figure();

subplot(3, length(alphas)+1, 1);
showgrey(F);
title("F");

subplot(3, length(alphas)+1, length(alphas)+2);
showfs(fft2(F));
title("Fhat");

for i=1:length(alphas)
    G = rot(F, alphas(i));
    subplot(3, length(alphas)+1, i+1);
    showgrey(G);
    title(strcat("G, ", int2str(alphas(i)), "º"));
    axis on;
    
    Ghat = fft2(G);
    subplot(3, length(alphas)+1, length(alphas)+1+i+1);
    showfs(Ghat);
    title("Ghat");
    
    Hhat = rot(fftshift(Ghat), -alphas(i));
    subplot(3, length(alphas)+1, 2*(length(alphas)+1)+i+1);
    showgrey(log(1 + abs(Hhat)))
    title("Hhat");
end


%% Part 1.8 - Phase & Magnitude - Question 13


a = 10^(-10);

subplot(3,3,1)
showgrey(phonecalc128);
title("Original");
subplot(3,3,4)
showgrey(pow2image(phonecalc128, a));
title("Phase info");
subplot(3,3,7)
showgrey(randphaseimage(phonecalc128));
title("Magnitude info");

subplot(3,3,2)
showgrey(few128);
title("Original");
subplot(3,3,5)
showgrey(pow2image(few128, a));
title("Phase info");
subplot(3,3,8)
showgrey(randphaseimage(few128));
title("Magnitude info");

subplot(3,3,3)
showgrey(nallo128);
title("Original");
subplot(3,3,6)
showgrey(pow2image(nallo128, a));
title("Phase info");
subplot(3,3,9)
showgrey(randphaseimage(nallo128));
title("Magnitude info");


%% Part 2.3 - Filtering procedure - Questions 14 to 16


t = [0.1 0.3 1.0 10.0 100.0];

subplot(3, length(t)+1, 1);
showgrey(deltafcn(128, 128));
title("Original");
for j=1:length(t)
    psf = gaussfft(deltafcn(128, 128), t(j));
    subplot(3, length(t)+1, 1+j);
    showgrey(psf);
    title(strcat("t=", num2str(t(j))));    
    var = variance(psf)
end

t = [1.0 4.0 16.0 64.0 256.0];

subplot(3, length(t)+1, length(t)+1+1);
showgrey(phonecalc128);
title("Original");
for j=1:length(t)
    psf = gaussfft(phonecalc128, t(j));
    subplot(3, length(t)+1, length(t)+1+1+j);
    showgrey(psf);
    title(strcat("t=", num2str(t(j))));
end

subplot(3, length(t)+1, 2*(length(t)+1)+1);
showgrey(nallo128);
title("Original");
for j=1:length(t)
    psf = gaussfft(nallo128, t(j));
    subplot(3, length(t)+1, 2*(length(t)+1)+1+j);
    showgrey(psf);
    title(strcat("t=", num2str(t(j))));
end


%% Part 3.1 - Smoothing of noisy data - Questions 17 & 18


office = office256;
add = gaussnoise(office, 16);
sap = sapnoise(office, 0.1, 255);

t = [0.7 1.0 1.3 1.7];
win = [2 3 4 5];
cutoff = [0.1 0.2 0.3 0.4];

subplot(3, length(t)+2, 1);
showgrey(office);
title("Original");

subplot(3, length(t)+2, 2);
showgrey(add);
title("Gaussian Noise");

for i=1:length(t)
    subplot(3, length(t)+2, 2+i);
    showgrey(gaussfft(add, t(i)));
    title(strcat("Gauss, t=", num2str(t(i))));
end

subplot(3, length(t)+2, (length(t)+2)+1);
showgrey(office);
title("Original");

subplot(3, length(t)+2, (length(t)+2)+2);
showgrey(add);
title("Gaussian Noise");

for i=1:length(win)
    subplot(3, length(t)+2, (length(t)+2)+2+i);
    showgrey(medfilt(add, win(i), win(i)));
    title(strcat("Med, winH=winW=", num2str(win(i))));
end

subplot(3, length(t)+2, 2*(length(t)+2)+1);
showgrey(office);
title("Original");

subplot(3, length(t)+2, 2*(length(t)+2)+2);
showgrey(add);
title("Gaussian Noise");

for i=1:length(cutoff)
    subplot(3, length(t)+2, 2*(length(t)+2)+2+i);
    showgrey(ideal(add, cutoff(i)));
    title(strcat("Ideal, cutoff=", num2str(cutoff(i))));
end

pause();

subplot(3, length(t)+2, 1);
showgrey(office);
title("Original");

subplot(3, length(t)+2, 2);
showgrey(sap);
title("Salt&Pepper Noise");

for i=1:length(t)
    subplot(3, length(t)+2, 2+i);
    showgrey(gaussfft(sap, t(i)));
    title(strcat("Gauss, t=", num2str(t(i))));
end

subplot(3, length(t)+2, (length(t)+2)+1);
showgrey(office);
title("Original");

subplot(3, length(t)+2, (length(t)+2)+2);
showgrey(sap);
title("Salt&Pepper Noise");

for i=1:length(win)
    subplot(3, length(t)+2, (length(t)+2)+2+i);
    showgrey(medfilt(sap, win(i), win(i)));
    title(strcat("Med, winH=winW=", num2str(win(i))));
end

subplot(3, length(t)+2, 2*(length(t)+2)+1);
showgrey(office);
title("Original");

subplot(3, length(t)+2, 2*(length(t)+2)+2);
showgrey(sap);
title("Salt&Pepper Noise");

for i=1:length(cutoff)
    subplot(3, length(t)+2, 2*(length(t)+2)+2+i);
    showgrey(ideal(sap, cutoff(i)));
    title(strcat("Ideal, cutoff=", num2str(cutoff(i))));
end


%% Part 3.2 - Smoothing and subsampling - Questions 19 & 20


% img = phonecalc256;
% gauss = img;
% id = img;
% N=5;
% 
% t=1.0;
% cutoff=0.2;
% 
% for i=1:N
%     if i>1 % generate subsampled versions
%         img = rawsubsample(img);
%         id = ideal(id, cutoff);
%         id = rawsubsample(id);
%         gauss = gaussfft(gauss, t);
%         gauss = rawsubsample(gauss);
%     end
%     subplot(3, N, i)
%     showgrey(img)
%     title("Subsampling");
%     subplot(3, N, i+N)
%     showgrey(gauss)
%     title(strcat("Gaussian, t=", num2str(t)));
%     subplot(3, N, 2*N+i)
%     showgrey(id)
%     title(strcat("Ideal, cutoff=", num2str(cutoff)));
% end
