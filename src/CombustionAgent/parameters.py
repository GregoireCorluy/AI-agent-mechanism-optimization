from pydantic import BaseModel, Field

class InputParameters(BaseModel):

    mechanism: str | None = Field(default = None, description = "Chemical mechanism") #give a list of possible mechanisms? Database for the mechanisms?
    fuel: str | None = Field(default = None, description = "Fuel used for the combustion simulation.")
    pressure_start: float | None = Field(default = None, description = "Lowerbound of the pressure at which the simulation is performed.")
    pressure_end: float | None = Field(default = None, description = "Upperbound of the pressure at which the simulation is performed.")
    pressure_unit: str | None = Field(default = None, description = "Unit of the pressure provided by the user. E.g., 'P' for Pascal. 'atm' for standard atmosphere, 'bar'.")
    temperature_start: float | None = Field(default = None, description = "Lowerbound of the temperature at which the simulation should be performed, keeping the number that is provided by the user. Do not convert to another unit.")
    temperature_end: float | None = Field(default = None, description = "Upperbound of the temperature at which the simulation should be performed, keeping the number that is provided by the user. Do not convert to another unit.")
    temperature_unit: str | None = Field(default = None, description = "Unit of the temperature provided by the user. E.g., 'K' for Kelvin. 'C' for Celsius. 'F' for Fahrenheit.")
    equivalence_ratio_start: float | None = Field(default = None, description = "Lowerbound of the equivalence ratio range")
    equivalence_ratio_end: float | None = Field(default = None, description = "Upperbound of the equivalence ratio range")
    target_species: str | None = Field(default = None, description = "Target species")