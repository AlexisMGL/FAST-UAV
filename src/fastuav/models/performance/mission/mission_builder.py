"""
Mission generator.
"""
import fastoad.api as oad
import openmdao.api as om
import numpy as np
from itertools import chain
from fastuav.constants import *
from fastuav.models.performance.mission.mission_definition.schema import MissionDefinition
from fastuav.models.performance.mission.route_builder import RouteBuilder


class GeneratorEnergySizing(om.ExplicitComponent):
    """
    Computes how much energy is provided by the thermal generator and how much
    must be retained in the batteries to cover a no-fuel contingency.
    """

    def initialize(self):
        self.options.declare("mission_name", default=None, types=str)
        self.options.declare("routes_list", default=None, types=list)

    def setup(self):
        mission_name = self.options["mission_name"]
        # Inputs describing the nominal sizing mission energy split
        self.add_input(f"mission:{mission_name}:energy:{FW_PROPULSION}", val=0.0, units="kJ")
        self.add_input(f"mission:{mission_name}:energy:{MR_PROPULSION}", val=0.0, units="kJ")
        self.add_input(f"mission:{mission_name}:main_route:cruise:energy", val=0.0, units="kJ")
        self.add_input(f"mission:{mission_name}:main_route:cruise:distance", val=0.0, units="m")
        self.add_input(f"mission:{mission_name}:main_route:hover:energy", val=0.0, units="kJ")

        # User inputs
        self.add_input("mission:sizing:nofuel", val=0.0, units="km")
        self.add_input(
            "models:generator:cons",
            val=0.0,
            units="kg/(kW*h)",
            desc="Fuel consumption in kg/kWh",
        )

        # Outputs used downstream
        self.add_output(f"mission:{mission_name}:generator:energy:{FW_PROPULSION}", units="kJ")
        self.add_output(f"mission:{mission_name}:generator:energy:{MR_PROPULSION}", units="kJ")
        self.add_output(f"mission:{mission_name}:nofuel:energy:{FW_PROPULSION}", units="kJ")
        self.add_output(f"mission:{mission_name}:nofuel:energy:{MR_PROPULSION}", units="kJ")
        self.add_output("data:weight:propulsion:generator:fuel:mass", units="kg")

    def setup_partials(self):
        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs):
        mission_name = self.options["mission_name"]
        nofuel_distance_m = inputs["mission:sizing:nofuel"] * 1000.0
        cruise_energy = inputs[f"mission:{mission_name}:main_route:cruise:energy"]
        cruise_distance = inputs[f"mission:{mission_name}:main_route:cruise:distance"]
        hover_energy = inputs[f"mission:{mission_name}:main_route:hover:energy"]
        cons = inputs["models:generator:cons"]

        energy_per_m = cruise_energy / cruise_distance if cruise_distance > 0 else 0.0
        nofuel_energy_fw = energy_per_m * nofuel_distance_m
        nofuel_energy_mr = hover_energy  # VTOL landing requirement

        mission_energy_fw = inputs[f"mission:{mission_name}:energy:{FW_PROPULSION}"]
        mission_energy_mr = inputs[f"mission:{mission_name}:energy:{MR_PROPULSION}"]

        gen_energy_fw = max(mission_energy_fw - nofuel_energy_fw, 0.0)
        gen_energy_mr = max(mission_energy_mr - nofuel_energy_mr, 0.0)
        total_gen_energy = gen_energy_fw + gen_energy_mr

        outputs[f"mission:{mission_name}:generator:energy:{FW_PROPULSION}"] = gen_energy_fw
        outputs[f"mission:{mission_name}:generator:energy:{MR_PROPULSION}"] = gen_energy_mr
        outputs[f"mission:{mission_name}:nofuel:energy:{FW_PROPULSION}"] = nofuel_energy_fw
        outputs[f"mission:{mission_name}:nofuel:energy:{MR_PROPULSION}"] = nofuel_energy_mr
        outputs["data:weight:propulsion:generator:fuel:mass"] = (
            total_gen_energy / 3600.0 * cons if cons > 0 else 0.0
        )


@oad.RegisterOpenMDAOSystem("fastuav.performance.mission")
class MissionBuilder(om.Group):
    """
    This class builds a mission from a provided definition.
    """

    def initialize(self):
        self.options.declare("file_path", default=None, types=str)

    def setup(self):
        file_path = self.options["file_path"]
        mission_dict = MissionDefinition(file_path)

        for mission_name, mission_definition in mission_dict[MISSION_DEFINITION_TAG].items():
            routes_list = []  # list of routes names
            propulsion_id_dict = {}  # list of propulsion systems used to complete the mission
            is_sizing = True if mission_name == SIZING_MISSION_TAG else False  # sizing mission flag

            # Create mission group
            mission_group = self.add_subsystem(mission_name, om.Group(), promotes=["*"])

            # Add routes to the mission group
            for route in mission_definition[PARTS_TAG]:
                _, route_name = tuple(*route.items())  # get route name
                route_definition = mission_dict[ROUTE_DEFINITION_TAG][route_name]  # get route definition
                routes_list.append(route_name)
                propulsion_id_dict[route_name] = RouteBuilder.get_propulsion_id_list(route_definition)
                # Add OpenMDAO subgroup to mission group
                mission_group.add_subsystem(route_name,
                                            RouteBuilder(mission_name=mission_name,
                                                         is_sizing=is_sizing,
                                                         route_name=route_name,
                                                         route_definition=route_definition),
                                            promotes=["*"])

            # Add mission component to sum up the routes calculations outputs
            mission_group.add_subsystem("mission",
                                        MissionComponent(mission_name=mission_name,
                                                         routes_list=routes_list,
                                                         propulsion_id_dict=propulsion_id_dict),
                                        promotes=["*"])
            # Optional generator / no-fuel sizing for hybrid missions
            if mission_name == SIZING_MISSION_TAG and {MR_PROPULSION, FW_PROPULSION}.issubset(
                set(chain(*propulsion_id_dict.values()))
            ) and "main_route" in routes_list:
                mission_group.add_subsystem(
                    "generator",
                    GeneratorEnergySizing(mission_name=mission_name, routes_list=routes_list),
                    promotes=["*"],
                )

            # Add constraint for sizing the battery capacity / energy to complete the mission
            mission_group.add_subsystem("constraints",
                                        MissionConstraints(mission_name=mission_name,
                                                           propulsion_id_dict=propulsion_id_dict),
                                        promotes=["*"])


class MissionComponent(om.ExplicitComponent):
    """
    This component computes the mission parameters (energy and duration) from the routes its made of.
    """

    def initialize(self):
        self.options.declare("mission_name", default=None, types=str)
        self.options.declare("routes_list", default=[], types=list)
        self.options.declare("propulsion_id_dict", default=None, types=dict)

    def setup(self):
        mission_name = self.options["mission_name"]
        routes_list = self.options["routes_list"]
        propulsion_id_dict = self.options["propulsion_id_dict"]

        for route_name in routes_list:
            for propulsion_id in propulsion_id_dict[route_name]:
                self.add_input("mission:%s:%s:energy:%s" % (mission_name, route_name, propulsion_id),
                               val=np.nan,
                               units="kJ")
                self.add_input("mission:%s:%s:duration:%s" % (mission_name, route_name, propulsion_id),
                               val=np.nan,
                               units="min")
            self.add_input("mission:%s:%s:energy" % (mission_name, route_name),
                           val=np.nan,
                           units="kJ")
            self.add_input("mission:%s:%s:duration" % (mission_name, route_name),
                           val=np.nan,
                           units="min")
            self.add_input("mission:%s:%s:cruise:distance" % (mission_name, route_name),
                           val=np.nan,
                           units="m")

        for propulsion_id in list(set(chain(*propulsion_id_dict.values()))):  # list of unique propulsion ids
            self.add_output("mission:%s:energy:%s" % (mission_name, propulsion_id), units="kJ")
            self.add_output("mission:%s:duration:%s" % (mission_name, propulsion_id), units="min")
        self.add_output("mission:%s:energy" % mission_name, units="kJ")
        self.add_output("mission:%s:duration" % mission_name, units="min")
        self.add_output("mission:%s:distance" % mission_name, units="m")

    def setup_partials(self):
        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs):
        mission_name = self.options["mission_name"]
        routes_list = self.options["routes_list"]
        propulsion_id_dict = self.options["propulsion_id_dict"]

        for propulsion_id in list(set(chain(*propulsion_id_dict.values()))):  # list of unique propulsion ids
            outputs["mission:%s:energy:%s" % (mission_name, propulsion_id)] = sum(
                inputs["mission:%s:%s:energy:%s" % (mission_name, route_name, propulsion_id)] for route_name in
                routes_list if propulsion_id in propulsion_id_dict[route_name])
            outputs["mission:%s:duration:%s" % (mission_name, propulsion_id)] = sum(
                inputs["mission:%s:%s:duration:%s" % (mission_name, route_name, propulsion_id)] for route_name in
                routes_list if propulsion_id in propulsion_id_dict[route_name])

        outputs["mission:%s:energy" % mission_name] = sum(
            inputs["mission:%s:%s:energy" % (mission_name, route_name)] for route_name in routes_list)

        outputs["mission:%s:duration" % mission_name] = sum(
            inputs["mission:%s:%s:duration" % (mission_name, route_name)] for route_name in routes_list)

        outputs["mission:%s:distance" % mission_name] = sum(
            inputs["mission:%s:%s:cruise:distance" % (mission_name, route_name)] for route_name in routes_list)


class MissionConstraints(om.ExplicitComponent):
    """
    This component computes the constraints associated with the battery energy required to perform a mission.
    """

    def initialize(self):
        self.options.declare("mission_name", default=None, types=str)
        self.options.declare("propulsion_id_dict", default=None, types=dict)

    def setup(self):
        mission_name = self.options["mission_name"]
        propulsion_id_dict = self.options["propulsion_id_dict"]
        for propulsion_id in list(set(chain(*propulsion_id_dict.values()))):  # list of unique propulsion ids
            self.add_input("mission:%s:energy:%s" % (mission_name, propulsion_id), val=np.nan, units="kJ")
            self.add_input("data:propulsion:%s:battery:energy" % propulsion_id, val=0.0, units="kJ")
            self.add_input("data:propulsion:%s:battery:DoD:max" % propulsion_id, val=0.8, units=None)
            self.add_input("mission:%s:generator:energy:%s" % (mission_name, propulsion_id), val=0.0, units="kJ")
            self.add_input("mission:%s:nofuel:energy:%s" % (mission_name, propulsion_id), val=0.0, units="kJ")
            self.add_output("optimization:constraints:mission:%s:energy:%s" % (mission_name, propulsion_id), units=None)
            self.add_output(
                "optimization:constraints:mission:%s:nofuel:%s" % (mission_name, propulsion_id), units=None
            )

    def setup_partials(self):
        self.declare_partials("*", "*", method="exact")

    def compute(self, inputs, outputs):
        mission_name = self.options["mission_name"]
        propulsion_id_dict = self.options["propulsion_id_dict"]

        for propulsion_id in list(set(chain(*propulsion_id_dict.values()))):  # list of unique propulsion ids
            E_mission = inputs["mission:%s:energy:%s" % (mission_name, propulsion_id)]
            E_gen = inputs["mission:%s:generator:energy:%s" % (mission_name, propulsion_id)]
            E_nofuel = inputs["mission:%s:nofuel:energy:%s" % (mission_name, propulsion_id)]
            E_bat = inputs["data:propulsion:%s:battery:energy" % propulsion_id]
            C_ratio = inputs["data:propulsion:%s:battery:DoD:max" % propulsion_id]
            usable_bat = E_bat * C_ratio
            energy_con = (usable_bat + E_gen - E_mission) / (usable_bat + E_gen) if (usable_bat + E_gen) > 0 else -1e6
            outputs["optimization:constraints:mission:%s:energy:%s" % (mission_name, propulsion_id)] = energy_con
            nofuel_con = (usable_bat - E_nofuel) / usable_bat if usable_bat > 0 else -1e6
            outputs["optimization:constraints:mission:%s:nofuel:%s" % (mission_name, propulsion_id)] = nofuel_con

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        mission_name = self.options["mission_name"]
        propulsion_id_dict = self.options["propulsion_id_dict"]

        for propulsion_id in list(set(chain(*propulsion_id_dict.values()))):  # list of unique propulsion ids
            E_mission = inputs["mission:%s:energy:%s" % (mission_name, propulsion_id)]
            E_gen = inputs["mission:%s:generator:energy:%s" % (mission_name, propulsion_id)]
            E_nofuel = inputs["mission:%s:nofuel:energy:%s" % (mission_name, propulsion_id)]
            E_bat = inputs["data:propulsion:%s:battery:energy" % propulsion_id]
            C_ratio = inputs["data:propulsion:%s:battery:DoD:max" % propulsion_id]
            usable_bat = E_bat * C_ratio
            denom = usable_bat + E_gen
            partials[
                "optimization:constraints:mission:%s:energy:%s" % (mission_name, propulsion_id),
                "mission:%s:energy:%s" % (mission_name, propulsion_id),
            ] = -1.0 / denom if denom > 0 else 0.0
            partials[
                "optimization:constraints:mission:%s:energy:%s" % (mission_name, propulsion_id),
                "data:propulsion:%s:battery:energy" % propulsion_id,
            ] = (C_ratio * E_mission) / (denom**2) if denom > 0 else 0.0
            partials[
                "optimization:constraints:mission:%s:energy:%s" % (mission_name, propulsion_id),
                "data:propulsion:%s:battery:DoD:max" % propulsion_id,
            ] = (E_bat * E_mission) / (denom**2) if denom > 0 else 0.0
            partials[
                "optimization:constraints:mission:%s:energy:%s" % (mission_name, propulsion_id),
                "mission:%s:generator:energy:%s" % (mission_name, propulsion_id),
            ] = E_mission / (denom**2) if denom > 0 else 0.0

            partials[
                "optimization:constraints:mission:%s:nofuel:%s" % (mission_name, propulsion_id),
                "mission:%s:nofuel:energy:%s" % (mission_name, propulsion_id),
            ] = -1.0 / usable_bat if usable_bat > 0 else 0.0
            partials[
                "optimization:constraints:mission:%s:nofuel:%s" % (mission_name, propulsion_id),
                "data:propulsion:%s:battery:energy" % propulsion_id,
            ] = (C_ratio * E_nofuel) / (usable_bat**2) if usable_bat > 0 else 0.0
            partials[
                "optimization:constraints:mission:%s:nofuel:%s" % (mission_name, propulsion_id),
                "data:propulsion:%s:battery:DoD:max" % propulsion_id,
            ] = (E_bat * E_nofuel) / (usable_bat**2) if usable_bat > 0 else 0.0
