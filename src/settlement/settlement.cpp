#include "settlement/settlement.h"

void SettlementModule::SetPrivousSettleItem(std::string& keySymbol, SettlementItem item) {
    boost::unique_lock<boost::shared_mutex> writeLock(this->mutex);
    this->previousSettleMap[keySymbol] = item;
}

bool SettlementModule::GetPrivousSettleItem(std::string& keySymbol, SettlementItem& item) {
    boost::shared_lock<boost::shared_mutex> readLock(this->mutex);
    auto it = this->previousSettleMap.find(keySymbol);
    if (it != this->previousSettleMap.end()) {
        item = it->second;
        return true;
    }
    return false;
}
