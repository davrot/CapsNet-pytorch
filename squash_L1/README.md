The squash functions is now replaced by the L1 norm.

For making this possible, I had to limit the output of the PrimaryCapsLayer, the CapsLayer, and the AgreementRouting to positive activites. Hence the CapsLayer got bias and relu layers were added. It looks like that there was no significant change in the classification performance. 




