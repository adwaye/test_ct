load('/home/adwaye/matlab_projects/test_CT/Figures/Buqo_problem_results.mat')


figure(30)
subplot(321);plot(l1reg), hold on, plot(hpd_constraint*ones(size(l1reg)),'r'),xlabel('it'), ylabel('l1 norm regulariser hpd'), 
subplot(322);plot(l2data), hold on, plot(epsilon*ones(size(l2data)),'r'), xlabel('it'), ylabel('l2 norm data fit')
subplot(323);plot(l2smooth), hold on, plot(theta*ones(size(l2smooth)),'r'),  xlabel('it'), ylabel('Mask energy')
subplot(324);plot(smooth_max), hold on, plot(tau*ones(size(smooth_max)),'r'),  xlabel('it'), ylabel('inpainting energy')
subplot(325);plot(dist2),   xlabel('it'), ylabel('Distance between xc and xs')