
def prem_density(radius):
    x = radius / float(6371e3)
    if radius > 6371e3:
        densprem=0
    elif radius <= 1221.5e3:
        densprem = 13.0885 - 8.8381 * x**2
    elif radius <= 3480e3:
        densprem = 12.5815 - 1.2638 * x - 3.6426 * x**2 - 5.5281 * x**3
    elif radius <= 3630.e3:
        densprem = 7.9565 - 6.4761 * x + 5.5283 * x**2 - 3.0807 * x**3
    elif radius <= 5600.e3:
        densprem = 7.9565 - 6.4761 * x + 5.5283 * x**2 - 3.0807 * x**3
    elif radius <= 5701.e3:
        densprem = 7.9565 - 6.4761 * x + 5.5283 * x**2 - 3.0807 * x**3
    elif radius <= 5771.e3:
        densprem = 5.3197 - 1.4836 * x
    elif radius <= 5971.e3:
        densprem = 11.2494 - 8.0298 * x
    elif radius <= 6151.e3:
        densprem = 7.1089 - 3.8045 * x
    elif radius <= 6291.e3:
        densprem = 2.6910 + 0.6924 * x
    elif radius <= 6346.e3:
        densprem = 2.6910 + 0.6924 * x
    elif radius <= 6356.e3:
        densprem = 2.9
    elif radius <= 6368.e3:
        densprem = 2.6
    else:
        densprem = 1.020
    return densprem * 1000
