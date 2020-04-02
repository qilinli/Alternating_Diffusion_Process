function A = label_correction(A, gnd, labeled_ind)
    for i = 1 : length(labeled_ind)
        for j = i+1 : length(labeled_ind)
            if gnd(labeled_ind(i)) ~= gnd(labeled_ind(j)) && A(labeled_ind(i), labeled_ind(j)) ~= 0        
                A(labeled_ind(i), labeled_ind(j)) = 0;
                A(labeled_ind(j), labeled_ind(i)) = 0;
            end
        end
    end