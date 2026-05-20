import { useState } from 'react'
import { Dialog, DialogContent, DialogHeader, DialogTitle } from './ui/dialog'
import worldBankLogo from '@/assets/The_World_Bank_logo.svg'

const DISCLAIMER_TEXT =
  'This tool is a product funded by The World Bank. However, the World Bank does not guarantee the accuracy or completeness of the data and does not assume responsibility for any errors or liability with respect to the use of the information or conclusions set forth. Results are indicative only. Exposure estimates and damage costs are derived from global hazard datasets and climate scenarios, which carry inherent uncertainties and may not capture local conditions. Vulnerability curves and replacement costs are approximations and should not be used as the sole basis for investment decisions or engineering design. Users are encouraged to complement this analysis with locally validated data and expert judgment.'

export default function DisclaimerDialog() {
  const [open, setOpen] = useState(true)

  return (
    <Dialog open={open} onOpenChange={() => {}} className="max-w-2xl">
      <DialogContent onClose={() => setOpen(false)}>
        <DialogHeader>
          <DialogTitle>Disclaimer</DialogTitle>
        </DialogHeader>
        <p className="text-sm text-gray-700 leading-relaxed pr-6">{DISCLAIMER_TEXT}</p>
        <div className="mt-6 flex justify-center">
          <img
            src={worldBankLogo}
            alt="The World Bank"
            className="h-12 w-auto"
          />
        </div>
      </DialogContent>
    </Dialog>
  )
}
