from urllib.parse import urljoin
import requests
from astropy import units as u
from crispy_forms.layout import Div, Layout
from dateutil.parser import parse
from django import forms
from django.conf import settings
from django.core.cache import cache

from tom_common.exceptions import ImproperCredentialsException
from tom_observations.facility import BaseRoboticObservationFacility, BaseRoboticObservationForm, get_service_class
from tom_targets.models import Target, REQUIRED_NON_SIDEREAL_FIELDS, REQUIRED_NON_SIDEREAL_FIELDS_PER_SCHEME

# Determine settings for this module.
try:
    SETTINGS = settings.FACILITIES["IAG"]
except (AttributeError, KeyError):
    SETTINGS = {
        "portal_url": "http://observe.monet.uni-goettingen.de",
        "archive_url": "http://archive.monet.uni-goettingen.de",
        "api_key": "",
        "archive_api_key": "",
    }

# Module specific settings.
PORTAL_URL = SETTINGS["portal_url"]
ARCHIVE_URL = SETTINGS["archive_url"]
SUCCESSFUL_OBSERVING_STATES = ["COMPLETED"]
FAILED_OBSERVING_STATES = ["WINDOW_EXPIRED", "CANCELED"]
TERMINAL_OBSERVING_STATES = SUCCESSFUL_OBSERVING_STATES + FAILED_OBSERVING_STATES

# Units of flux and wavelength for converting to Specutils Spectrum1D objects
FLUX_CONSTANT = (1e-15 * u.erg) / (u.cm**2 * u.second * u.angstrom)
WAVELENGTH_UNITS = u.angstrom

# FITS header keywords used for data processing
FITS_FACILITY_KEYWORD = "ORIGIN"
FITS_FACILITY_KEYWORD_VALUE = "IAG"
FITS_FACILITY_DATE_OBS_KEYWORD = "DATE-OBS"

# Functions needed specifically for IAG
# Helpers for IAG fields
ipp_value_help = "Value between 0.5 to 2.0."


def make_request(*args, **kwargs):
    response = requests.request(*args, **kwargs)
    if 400 <= response.status_code < 500:
        raise ImproperCredentialsException("IAG: " + str(response.content))
    response.raise_for_status()
    return response


def get_instruments():
    cached_instruments = cache.get("monet_instruments")

    if not cached_instruments:
        response = make_request(
            "GET", PORTAL_URL + "/api/instruments/", headers={"Authorization": "Token {0}".format(SETTINGS["api_key"])}
        )
        cached_instruments = {k: v for k, v in response.json().items() if "SOAR" not in k}
        cache.set("monet_instruments", cached_instruments)

    return cached_instruments


class IAGBaseForm(forms.Form):
    ipp_value = forms.FloatField()
    exposure_count = forms.IntegerField(min_value=1)
    exposure_time = forms.FloatField(min_value=0.1)
    max_airmass = forms.FloatField()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields["proposal"] = forms.ChoiceField(choices=self.proposal_choices())
        self.fields["filter"] = forms.ChoiceField(choices=self.filter_choices())
        self.fields["instrument_type"] = forms.ChoiceField(choices=self.instrument_choices(), label="Instrument")

    def instrument_choices(self):
        return sorted([(k, v["name"]) for k, v in get_instruments().items()], key=lambda inst: inst[1])

    def filter_choices(self):
        return sorted(
            set(
                [
                    (f["code"], f["name"])
                    for ins in get_instruments().values()
                    for f in ins["optical_elements"].get("filters", []) + ins["optical_elements"].get("slits", [])
                ]
            ),
            key=lambda filter_tuple: filter_tuple[1],
        )

    def readout_choices(self):
        return sorted(
            [(f["code"], f["name"]) for ins in get_instruments().values() for f in ins["modes"]["readout"]["modes"]]
        )

    def proposal_choices(self):
        response = make_request(
            "GET", PORTAL_URL + "/api/profile/", headers={"Authorization": "Token {0}".format(SETTINGS["api_key"])}
        )
        choices = []
        for p in response.json()["proposals"]:
            if p["current"]:
                choices.append((p["id"], "{} ({})".format(p["title"], p["id"])))
        return choices


class IAGBaseObservationForm(BaseRoboticObservationForm, IAGBaseForm):
    """
    The IAGBaseObservationForm provides the base set of utilities to construct an observation at IAG.
    While the forms that inherit from it provide a subset of instruments and filters, the
    IAGBaseObservationForm presents the user with all of the instrument and filter options that the facility has to
    offer.

    IAG uses the LCO observation portal, so the used API is identical.
    """

    name = forms.CharField()
    ipp_value = forms.FloatField(label="IPP factor", min_value=0.5, max_value=2, initial=1.05, help_text=ipp_value_help)
    start = forms.CharField(widget=forms.TextInput(attrs={"type": "date"}))
    end = forms.CharField(widget=forms.TextInput(attrs={"type": "date"}))
    exposure_count = forms.IntegerField(min_value=1)
    exposure_time = forms.FloatField(min_value=0.1, widget=forms.TextInput(attrs={"placeholder": "Seconds"}))
    repeat_duration = forms.FloatField(required=False, label="Repeat duration [s]")
    max_airmass = forms.FloatField(initial=2)
    min_lunar_distance = forms.IntegerField(min_value=0, label="Minimum Lunar Distance", required=False)
    observation_mode = forms.ChoiceField(choices=[("NORMAL", "Normal")])

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields["readout_mode"] = forms.ChoiceField(choices=self.readout_choices())
        self.helper.layout = Layout(self.common_layout, self.layout(), self.button_layout())

    def layout(self):
        return Div(
            Div(
                Div("name", css_class="col"),
                Div("proposal", css_class="col"),
                css_class="form-row",
            ),
            Div(
                Div("ipp_value", css_class="col"),
                Div("observation_mode", css_class="col"),
                css_class="form-row",
            ),
            Div(
                Div("instrument_type", css_class="col"),
                Div("filter", css_class="col"),
                css_class="form-row",
            ),
            Div(
                Div("exposure_count", css_class="col"),
                Div("exposure_time", css_class="col"),
                css_class="form-row",
            ),
            Div(
                Div("max_airmass", css_class="col"),
                Div("min_lunar_distance", css_class="col"),
                css_class="form-row",
            ),
            Div(
                Div("repeat_duration", css_class="col"),
                css_class="form-row",
            ),
        )

    def clean_start(self):
        start = self.cleaned_data["start"]
        return parse(start).isoformat()

    def clean_end(self):
        end = self.cleaned_data["end"]
        return parse(end).isoformat()

    def is_valid(self):
        super().is_valid()
        obs_module = get_service_class(self.cleaned_data["facility"])
        errors = obs_module().validate_observation(self.observation_payload())
        if errors:
            self.add_error(None, self._flatten_error_dict(errors))
        return not errors

    def _flatten_error_dict(self, error_dict):
        non_field_errors = []
        for k, v in error_dict.items():
            if type(v) == list:
                for i in v:
                    if type(i) == str:
                        if k in self.fields:
                            self.add_error(k, i)
                        else:
                            non_field_errors.append("{}: {}".format(k, i))
                    if type(i) == dict:
                        non_field_errors.append(self._flatten_error_dict(i))
            elif type(v) == str:
                if k in self.fields:
                    self.add_error(k, v)
                else:
                    non_field_errors.append("{}: {}".format(k, v))
            elif type(v) == dict:
                non_field_errors.append(self._flatten_error_dict(v))

        return non_field_errors

    def instrument_to_type(self, instrument_type):
        return "EXPOSE"

    def _build_target_fields(self):
        target = Target.objects.get(pk=self.cleaned_data["target_id"])
        target_fields = {
            "name": target.name,
        }
        if target.type == Target.SIDEREAL:
            target_fields["type"] = "ICRS"
            target_fields["ra"] = target.ra
            target_fields["dec"] = target.dec
            target_fields["proper_motion_ra"] = target.pm_ra
            target_fields["proper_motion_dec"] = target.pm_dec
            target_fields["epoch"] = target.epoch
        elif target.type == Target.NON_SIDEREAL:
            target_fields["type"] = "ORBITAL_ELEMENTS"
            # Mapping from TOM field names to LCO API field names, for fields
            # where there are differences
            field_mapping = {
                "inclination": "orbinc",
                "lng_asc_node": "longascnode",
                "arg_of_perihelion": "argofperih",
                "semimajor_axis": "meandist",
                "mean_anomaly": "meananom",
                "mean_daily_motion": "dailymot",
                "epoch_of_elements": "epochofel",
                "epoch_of_perihelion": "epochofperih",
            }
            # The fields to include in the payload depend on the scheme. Add
            # only those that are required
            fields = REQUIRED_NON_SIDEREAL_FIELDS + REQUIRED_NON_SIDEREAL_FIELDS_PER_SCHEME[target.scheme]
            for field in fields:
                lco_field = field_mapping.get(field, field)
                target_fields[lco_field] = getattr(target, field)

        return target_fields

    def _build_instrument_config(self):
        instrument_config = {
            "exposure_count": self.cleaned_data["exposure_count"],
            "exposure_time": self.cleaned_data["exposure_time"],
            "mode": self.cleaned_data["readout_mode"],
            "optical_elements": {"filter": self.cleaned_data["filter"]},
        }

        return [instrument_config]

    def _build_acquisition_config(self):
        acquisition_config = {}

        return acquisition_config

    def _build_guiding_config(self):
        guiding_config = {}

        return guiding_config

    def _build_configuration(self):
        # build config
        cfg = {
            "type": "EXPOSE",
            "instrument_type": self.cleaned_data["instrument_type"],
            "target": self._build_target_fields(),
            "instrument_configs": self._build_instrument_config(),
            "acquisition_config": self._build_acquisition_config(),
            "guiding_config": self._build_guiding_config(),
            "constraints": {
                "max_airmass": self.cleaned_data["max_airmass"],
                "min_lunar_distance": self.cleaned_data["min_lunar_distance"],
            },
        }

        # EXPOSE or REPEAT_EXPOSE?
        if self.cleaned_data["repeat_duration"]:
            cfg["type"] = "REPEAT_EXPOSE"
            cfg["repeat_duration"] = self.cleaned_data["repeat_duration"]

        # return it
        return cfg

    def _build_location(self):
        return {"telescope_class": get_instruments()[self.cleaned_data["instrument_type"]]["class"]}

    def observation_payload(self):
        payload = {
            "name": self.cleaned_data["name"],
            "proposal": self.cleaned_data["proposal"],
            "ipp_value": self.cleaned_data["ipp_value"],
            "operator": "SINGLE",
            "observation_type": self.cleaned_data["observation_mode"],
            "requests": [
                {
                    "configurations": [self._build_configuration()],
                    "windows": [{"start": self.cleaned_data["start"], "end": self.cleaned_data["end"]}],
                    "location": self._build_location(),
                }
            ],
        }

        return payload


class IAGImagingObservationForm(IAGBaseObservationForm):
    acquisition = forms.ChoiceField(choices=(("ON", "On"), ("OFF", "Off")), required=True, initial="ON")
    guiding = forms.ChoiceField(choices=(("ON", "On"), ("OFF", "Off")), required=True, initial="OFF")

    def __init__(self, *args, **kwargs):
        IAGBaseObservationForm.__init__(self, *args, **kwargs)

    @property
    def _instrument(self):
        # in initials?
        if "instrument_type" in self.initial:
            i = get_instruments()
            return i[self.initial["instrument_type"]] if self.initial["instrument_type"] in i else None
        return None

    def instrument_choices(self):
        # get instruments
        instruments = get_instruments()

        if "instrument" in self.initial:
            # given instrument
            if self.initial["instrument"] in instruments:
                return [(self.initial["instrument_type"], instruments[self.initial["instrument_type"]]["name"])]
            else:
                return []
        else:
            # return all
            return sorted(
                [(k, v["name"]) for k, v in get_instruments().items() if "IMAGE" in v["type"]], key=lambda inst: inst[1]
            )

    def filter_choices(self):
        return sorted(
            set(
                [
                    (f["code"], f["name"])
                    for ins in get_instruments().values()
                    for f in ins["optical_elements"].get("filters", [])
                ]
            ),
            key=lambda filter_tuple: filter_tuple[1],
        )

    def _build_acquisition_config(self):
        acquisition_config = {"mode": self.cleaned_data["acquisition"]}

        return acquisition_config

    def _build_guiding_config(self):
        guiding_config = {"mode": self.cleaned_data["guiding"]}

        return guiding_config

    def layout(self):
        return Div(
            Div(
                Div("name", css_class="col"),
                Div("proposal", css_class="col"),
                css_class="form-row",
            ),
            Div(
                Div("observation_mode", css_class="col"),
                Div("ipp_value", css_class="col"),
                css_class="form-row",
            ),
            Div(
                Div("instrument_type", css_class="col"),
                css_class="form-row",
            ),
            Div(
                Div("exposure_count", css_class="col"),
                Div("exposure_time", css_class="col"),
                css_class="form-row",
            ),
            Div(
                Div("repeat_duration", css_class="col"),
                Div(None, css_class="col"),
                css_class="form-row",
            ),
            Div(
                Div("readout_mode", css_class="col"),
                Div("filter", css_class="col"),
                css_class="form-row",
            ),
            Div(
                Div("max_airmass", css_class="col"),
                Div("min_lunar_distance", css_class="col"),
                css_class="form-row",
            ),
            Div(
                Div("acquisition", css_class="col"),
                Div("guiding", css_class="col"),
                css_class="form-row",
            ),
            Div(
                Div("start", css_class="col"),
                Div("end", css_class="col"),
                css_class="form-row",
            ),
        )

    def filter_choices(self):
        return sorted(
            set([(f["code"], f["name"]) for f in self._instrument["optical_elements"].get("filters", [])]),
            key=lambda filter_tuple: filter_tuple[1],
        )

    def readout_choices(self):
        return sorted([(f["code"], f["name"]) for f in self._instrument["modes"]["readout"]["modes"]])


class MonetSouthImagingObservationForm(IAGImagingObservationForm):
    INSTRUMENT = "1M2 FLI230"

    @property
    def _instrument(self):
        return get_instruments()[MonetSouthImagingObservationForm.INSTRUMENT]

    def instrument_choices(self):
        return [(MonetSouthImagingObservationForm.INSTRUMENT, self._instrument["name"])]


class MonetNorthImagingObservationForm(IAGImagingObservationForm):
    INSTRUMENT = "1M2 SBIG8300"

    @property
    def _instrument(self):
        return get_instruments()[MonetNorthImagingObservationForm.INSTRUMENT]

    def instrument_choices(self):
        return [(MonetNorthImagingObservationForm.INSTRUMENT, self._instrument["name"])]


class IAG50ImagingObservationForm(IAGImagingObservationForm):
    INSTRUMENT = "0M5 SBIG6303E"

    @property
    def _instrument(self):
        return get_instruments()[IAG50ImagingObservationForm.INSTRUMENT]

    def instrument_choices(self):
        return [(IAG50ImagingObservationForm.INSTRUMENT, self._instrument["name"])]


class IAGFacility(BaseRoboticObservationFacility):
    """
    The ``IAGFacility`` is the interface to the Observation Portal of the Institute for Astrophysics in Göttingen.
    """

    name = "IAG"
    observation_forms = {
        "MONET_SOUTH_IMAGING": MonetSouthImagingObservationForm,
        "MONET_NORTH_IMAGING": MonetNorthImagingObservationForm,
        "IAG50_IMAGING": IAG50ImagingObservationForm,
    }
    # The SITES dictionary is used to calculate visibility intervals in the
    # planning tool. All entries should contain latitude, longitude, elevation
    # and a code.
    # TODO: Flip sitecode and site name
    # TODO: Why is tlv not represented here?
    SITES = {
        "Sutherland": {"sitecode": "cpt", "latitude": -32.38, "longitude": 20.81, "elevation": 1804},
        "McDonald": {"sitecode": "elp", "latitude": 30.679, "longitude": -104.015, "elevation": 2027},
        "Göttingen": {"sitecode": "goe", "latitude": 51.560583, "longitude": 9.944333, "elevation": 201},
    }

    def get_form(self, observation_type):
        try:
            return self.observation_forms[observation_type]
        except KeyError:
            return IAGBaseObservationForm

    def submit_observation(self, observation_payload):
        response = make_request(
            "POST", PORTAL_URL + "/api/requestgroups/", json=observation_payload, headers=self._portal_headers()
        )
        return [r["id"] for r in response.json()["requests"]]

    def validate_observation(self, observation_payload):
        response = make_request(
            "POST",
            PORTAL_URL + "/api/requestgroups/validate/",
            json=observation_payload,
            headers=self._portal_headers(),
        )
        return response.json()["errors"]

    def cancel_observation(self, observation_id):
        response = make_request(
            "POST", f"{PORTAL_URL}/api/requestgroups/{observation_id}/cancel/", headers=self._portal_headers()
        )
        return response.json()["errors"]

    def get_observation_url(self, observation_id):
        return PORTAL_URL + "/requests/" + observation_id

    def get_flux_constant(self):
        return FLUX_CONSTANT

    def get_wavelength_units(self):
        return WAVELENGTH_UNITS

    def get_date_obs_from_fits_header(self, header):
        return header.get(FITS_FACILITY_DATE_OBS_KEYWORD, None)

    def is_fits_facility(self, header):
        """
        Returns True if the 'ORIGIN' keyword is in the given FITS header and contains the value 'IAG', False
        otherwise.

        :param header: FITS header object
        :type header: dictionary-like

        :returns: True if header matches IAG, False otherwise
        :rtype: boolean
        """
        return FITS_FACILITY_KEYWORD_VALUE == header.get(FITS_FACILITY_KEYWORD, None)

    def get_start_end_keywords(self):
        return ("start", "end")

    def get_terminal_observing_states(self):
        return TERMINAL_OBSERVING_STATES

    def get_failed_observing_states(self):
        return FAILED_OBSERVING_STATES

    def get_observing_sites(self):
        return self.SITES

    def get_facility_weather_urls(self):
        """
        `facility_weather_urls = {'code': 'XYZ', 'sites': [ site_dict, ... ]}`
        where
        `site_dict = {'code': 'XYZ', 'weather_url': 'http://path/to/weather'}`
        """
        return {}

    def get_facility_status(self):
        """
        Get the telescope_states and simply
        transform the returned JSON into the following dictionary hierarchy
        for use by the facility_status.html template partial.
        """
        return {}

    def get_observation_status(self, observation_id):
        response = make_request(
            "GET", PORTAL_URL + "/api/requests/{0}".format(observation_id), headers=self._portal_headers()
        )
        state = response.json()["state"]

        response = make_request(
            "GET", PORTAL_URL + "/api/requests/{0}/observations/".format(observation_id), headers=self._portal_headers()
        )
        blocks = response.json()
        current_block = None
        for block in blocks:
            if block["state"] == "COMPLETED":
                current_block = block
                break
            elif block["state"] == "PENDING":
                current_block = block
        if current_block:
            scheduled_start = current_block["start"]
            scheduled_end = current_block["end"]
        else:
            scheduled_start, scheduled_end = None, None

        return {"state": state, "scheduled_start": scheduled_start, "scheduled_end": scheduled_end}

    def data_products(self, observation_id, product_id=None):
        products = []
        for frame in self.archive_frames(observation_id, product_id):
            products.append(
                {
                    "id": frame["id"],
                    "filename": frame["basename"] + ".fits.gz",
                    "created": parse(frame["DATE_OBS"]),
                    "url": urljoin(ARCHIVE_URL, frame["url"]),
                    "imagetype": frame["OBSTYPE"],
                    "rlevel": frame["RLEVEL"],
                }
            )
        return products

    # The following methods are used internally by this module
    # and should not be called directly from outside code.

    def _portal_headers(self):
        if SETTINGS.get("api_key"):
            return {"Authorization": "Token {0}".format(SETTINGS["api_key"])}
        else:
            return {}

    def archive_headers(self):
        if SETTINGS.get("archive_api_key"):
            return {"Authorization": "Token {0}".format(SETTINGS["archive_api_key"])}
        else:
            return {}

    def archive_frames(self, observation_id, product_id=None):
        # todo save this key somewhere
        frames = []
        if product_id:
            response = make_request(
                "GET", ARCHIVE_URL + "/frames/{0}/".format(product_id), headers=self.archive_headers()
            )
            frames = [response.json()]

        else:
            # probably need to make multiple requests
            self._list_archive_frames(frames, observation_id)

        return frames

    def _list_archive_frames(self, frames, observation_id, offset=0, limit=500):
        # do request
        response = make_request(
            "GET",
            ARCHIVE_URL + "/frames/",
            params={"REQNUM": observation_id, "offset": offset, "limit": limit},
            headers=self.archive_headers(),
        )

        # get response
        res = response.json()

        # append frames
        frames.extend(res["results"])

        # need more?
        if len(frames) < res["count"]:
            self._list_archive_frames(frames, observation_id, offset + limit, limit)
