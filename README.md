# Daily Novel Coronavirus (COVID-19) REPORT

## Report Generator for USA / California / Santa Clara



This tool will generate a report about the COVID-19 spread in the USA, California, and Santa Clara.

It can be EASILY modified to cover other countries, states, counties.

**NOTE:** On FIRST run it will take like a minute to download the data from GIT.

### The REPORT:

This tool generates a REPORT covering the USA, California, and Santa Clara and stores it as a PDF.

The COVER letter is a HTML document, which can be easily modified to accommodate different languages and country settings.

The PDF is then send out via GMAIL. (Configuration instructions can be found here: https://stackoverflow.com/questions/37201250/sending-email-via-gmail-python)

The report and all other DATA can be found under the "Corona Data" directory, which is located in the directory from where you started the report tool from.

### The DATA:

We use the COVID-19 Data Repository by the Center for Systems Science and Engineering (CSSE) at Johns Hopkins University

https://github.com/CSSEGISandData/COVID-19

We ONLY use the # of death to extrapolate all other information, as this data is the most accurate. The number of infected is hugely under reported, as we did NOT have enough testing capacity as well as not being allowed to be tested in the first couple months.

### The Data PROCESSING:

We use KAHN's formula to calculate estimated number of infected people.

https://youtu.be/mCa0JXEwDEk

#### Specifically he uses:

| Calculations based on    | Value                                     |
| ------------------------ | ----------------------------------------- |
| estimated_time_to_die    | 17.33      (days from infection to death) |
| estimated_days_to_double | 6.18                                      |
| estimated_mortality_rate | 1%                                        |

These numbers can be adjusted for your region in the first few lines of the code.

### Corona_Watch.py:

This is not 100% working yet ... it monitors for changes in the Git-PULL of the Johns Hopkins data and runs a report once a change is detected. Unfortunately there is still a little but to make it stop once in a while ...

### Security HASHES:

Name: Corona.py
Size: 79138 bytes (77 KiB)
SHA256: 25225303CF4500325B188726AB72ADF89728EC8607BAAB2DE0264788E108E0F4

Name: Corona_Cover.html
Size: 94049 bytes (91 KiB)
SHA256: EFB638139867838DAA16EA88F40E78B46482964C13AA98670C8A1A51D5712635
