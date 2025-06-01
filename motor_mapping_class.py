import numpy as np

# class MotorMapping:
#     def __init__(self, coeffs1, coeffs2, coeffs_section2, coeffs_section3):
#         """
#         coeffs1: coefficients for motor1 (polynomial from first sorted_total_bending set)
#         coeffs2: coefficients for motor2 (polynomial from first sorted_total_bending set)
#         coeffs_section2: coefficients for the additional mapping (from a different sorted_total_bending)
#         """
#         self.motor1_section1_fn = np.poly1d(coeffs1)
#         self.motor2_section1_fn = np.poly1d(coeffs2)
#         self.motor_section2_fn = np.poly1d(coeffs_section2)
#         self.motor_section3_fn = np.poly1d(coeffs_section3)

#     def __call__(self, bending1, bending2, bending3):
#         """
#         bending1: input for motor1 and motor2 mapping (can be scalar or array)
#         bending2: input for the extra mapping (can be scalar or array)
#         Returns:
#             motor1, motor2, section2
#         """
#         # Evaluate the polynomials
#         motor1_section1 = self.motor1_section1_fn(bending1)
#         motor2_section1 = self.motor2_section1_fn(bending1)
#         section2 = self.motor_section2_fn(bending2)
#         section3 = self.motor_section3_fn(bending3)
        
#         return int(motor1_section1), int(motor2_section1), int(section2), int(section3)
    
class MotorMapping:
    def __init__(self, coeffs1, coeffs2, coeffs_section2, coeffs_section3):
        """
        Initialize motor mapping with polynomial coefficients and also create inverse interpolation functions.
        """
        # Forward mappings
        self.motor1_section1_fn = np.poly1d(coeffs1)
        self.motor2_section1_fn = np.poly1d(coeffs2)
        self.motor_section2_fn = np.poly1d(coeffs_section2)
        self.motor_section3_fn = np.poly1d(coeffs_section3)

        # Inverse mappings (motor position -> total bending)
        self.inv_motor1_fn = self._generate_inverse(self.motor1_section1_fn)
        self.inv_motor2_fn = self._generate_inverse(self.motor2_section1_fn)
        self.inv_section2_fn = self._generate_inverse(self.motor_section2_fn)
        self.inv_section3_fn = self._generate_inverse(self.motor_section3_fn)

    def __call__(self, bending1, bending2, bending3):
        """
        bending1: input for motor1 and motor2 mapping (can be scalar or array)
        bending2: input for section2
        bending3: input for section3
        Returns:
            motor1, motor2, section2, section3 (all as int)
        """
        motor1 = int(self.motor1_section1_fn(bending1))
        motor2 = int(self.motor2_section1_fn(bending1))
        section2 = int(self.motor_section2_fn(bending2))
        section3 = int(self.motor_section3_fn(bending3))
        return motor1, motor2, section2, section3

    def inverse(self, motor1_pos, motor2_pos, section2_pos, section3_pos):
        """
        Returns bending1, bending2, bending3 given motor positions.
        """
        bending1_from_motor1 = float(self.inv_motor1_fn(motor1_pos))
        bending1_from_motor2 = float(self.inv_motor2_fn(motor2_pos))
        bending2 = float(self.inv_section2_fn(section2_pos))
        bending3 = float(self.inv_section3_fn(section3_pos))
        return bending1_from_motor1, bending1_from_motor2, bending2, bending3

    def _generate_inverse(self, poly_fn, num_points=1000, domain=(0, 1)):
        """
        Generate inverse using interpolation.
        poly_fn: polynomial function to invert
        domain: range of bending input to sample
        """
        x = np.linspace(domain[0], domain[1], num_points)
        y = poly_fn(x)

        # Ensure the interpolation is monotonic by sorting
        sorted_indices = np.argsort(y)
        y_sorted = y[sorted_indices]
        x_sorted = x[sorted_indices]

        return interp1d(y_sorted, x_sorted, bounds_error=False, fill_value="extrapolate")