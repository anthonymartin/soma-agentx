# AGPL-3.0 Compliance Guide

This document provides guidance on how to comply with the GNU Affero General Public License v3.0 (AGPL-3.0) when using, modifying, or distributing the SOMA (Self-Organizing Memory Architecture) project.

## Overview of AGPL-3.0

The AGPL-3.0 is a copyleft license that requires anyone who distributes the software or modified versions of it to make the source code available under the same license. The AGPL-3.0 extends this requirement to include users who interact with the software over a network.

## Key Requirements

When using SOMA, you must comply with the following key requirements:

### 1. Source Code Distribution

If you distribute copies of SOMA, whether modified or unmodified, you must:
- Provide the complete corresponding source code
- Include a copy of the AGPL-3.0 license
- Preserve copyright notices and license statements in the code

### 2. Network Use Provision (Section 13)

If you modify SOMA and run it on a server that users can interact with over a network (e.g., as a web application), you must:
- Provide a way for users to download the source code of your modified version
- This is the key difference between the AGPL and the regular GPL

### 3. License Notices

You must keep intact all notices that refer to the AGPL-3.0 license and the absence of any warranty.

### 4. User Notices

You must make sure that users who interact with SOMA can access the license text and understand their rights.

## Practical Compliance Steps

### For Developers

1. **Keep the License File**: Always include the `LICENSE` file with any distribution.

2. **Maintain Copyright Headers**: All source files should include the copyright and license notice. Use the provided `add_copyright_headers.py` script to add headers to new files:
   ```bash
   ./add_copyright_headers.py --dir path/to/your/files
   ```

3. **Document Changes**: When modifying the code, document your changes clearly.

4. **Network Applications**: If you deploy a modified version of SOMA as a network service, add a feature to allow users to download the source code. For example:
   - Add a "Source Code" link in the user interface
   - Provide an API endpoint that serves the source code
   - Document how users can request and receive the source code

### For Users

1. **Understand Your Rights**: As a user of SOMA, you have the right to:
   - Use the software for any purpose
   - Study how the software works and modify it
   - Redistribute the software
   - Distribute your modifications

2. **Network Use**: If you interact with a modified version of SOMA over a network, you have the right to receive its source code.

## Common Compliance Scenarios

### Scenario 1: Internal Use Only

If you're only using SOMA internally within your organization and not distributing it or making it available over a network to external users, you don't need to share your modifications.

### Scenario 2: Distribution of Modified Version

If you distribute a modified version of SOMA:
1. Make your source code available under AGPL-3.0
2. Include the original license and copyright notices
3. Document your changes
4. Provide clear instructions on how to obtain the complete source code

### Scenario 3: Network Service

If you run a modified version of SOMA on a server:
1. Provide a way for users to download the source code
2. Include a notice informing users of their right to the source code
3. Ensure the download includes the complete source code of your modified version

## Additional Resources

- [Full text of the AGPL-3.0 license](https://www.gnu.org/licenses/agpl-3.0.en.html)
- [GNU AGPL-3.0 FAQ](https://www.gnu.org/licenses/gpl-faq.html)
- [Free Software Foundation](https://www.fsf.org/)

## Contact

If you have questions about AGPL-3.0 compliance for this project, please contact Cadenzai, Inc.
